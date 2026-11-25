"""Unity 工具的 C# 靜態檢查。這個專案的 C# 沒辦法在 CI 上用 Unity 編譯，所以至少用真正的 C# 文法分析一遍：
  ・語法錯誤（v1.12.0／v1.13.0 就是一行插錯位置、把 if…else if 拆開，整個工具無法編譯）
  ・區域變數重複宣告（CS0128）、巢狀 scope 和外層同名（CS0136）
  ・呼叫檔案裡自己宣告的方法時參數個數不對、DesktopRig.X 的 X 沒有宣告或不是 public
抓不到的：型別錯誤、Unity／VRChat SDK 的 API 用錯。需要 tree-sitter、tree-sitter-c-sharp（CI 會裝；本機沒有時跳過）。
執行：python tests/test_csharp.py"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ['unity/Desktop/DesktopFBTDesktop.cs', 'unity/ParamCompressor/Editor/DesktopFBTParamCompressorPass.cs',
         'unity/ParamCompressor/Runtime/DesktopFBTParamCompressor.cs', 'unity/DesktopFBTBvhImporter.cs']
SCOPES = {'block', 'for_statement', 'foreach_statement', 'using_statement', 'switch_section', 'lambda_expression',
          'anonymous_method_expression', 'catch_clause', 'local_function_statement'}
OUTER = ('method_declaration', 'constructor_declaration', 'accessor_declaration', 'class_declaration')
INHERITED = {'InPhase'}                      # 從基底類別來的方法（NDMF 的 Plugin）


def parser():
    try:
        import tree_sitter_c_sharp as cs
        from tree_sitter import Language, Parser
    except ImportError:
        return None
    return Parser(Language(cs.language()))


def text(n, src):
    return src[n.start_byte:n.end_byte].decode('utf-8', 'replace')


KEYWORDS = set("""abstract as base bool break byte case catch char checked class const continue decimal default delegate do double else enum
event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is lock long namespace new null object
operator out override params private protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch this
throw true try typeof uint ulong unchecked unsafe ushort using virtual void volatile while""".split())


def syntax(tree, src):
    out = []

    def walk(n):
        if n.type == 'ERROR' or n.is_missing:
            out.append((n.start_point[0] + 1, 'syntax', text(n, src)[:60]))
        elif n.type == 'identifier' and text(n, src) in KEYWORDS:
            # 文法分析器比編譯器寬鬆：落單的 else 會被它當成「回傳型別叫 else 的函式」。保留字被當成名稱 = 一定是語法錯誤
            out.append((n.start_point[0] + 1, 'syntax', '保留字被當成名稱：' + text(n, src)))
        for c in n.children:
            walk(c)
    walk(tree.root_node)
    return out


def declared_here(n, src):
    f = None
    if n.type == 'variable_declarator':
        f = n.child_by_field_name('name') or next((c for c in n.children if c.type == 'identifier'), None)
    elif n.type in ('parameter', 'declaration_expression', 'catch_declaration'):
        f = n.child_by_field_name('name')
    elif n.type == 'implicit_parameter':
        f = n
    return [(text(f, src), f.start_point[0] + 1)] if f is not None else []


def scopes(tree, src):
    out = []

    def collect(scope):
        own, kids = [], []
        skip = None
        if scope.type == 'foreach_statement':
            left, skip = scope.child_by_field_name('left'), scope.child_by_field_name('right')
            if left is not None and left.type == 'identifier':
                own.append((text(left, src), left.start_point[0] + 1))

        def walk(m):
            for c in m.children:
                if c.type in OUTER or (skip is not None and c == skip):
                    continue
                if c.type in SCOPES:
                    kids.append(c)
                    if c.type == 'foreach_statement':          # 「in」後面的運算式屬於外面這一層
                        right = c.child_by_field_name('right')
                        if right is not None:
                            walk_one(right)
                    continue
                own.extend(declared_here(c, src))
                walk(c)

        def walk_one(c):
            if c.type in SCOPES:
                kids.append(c)
            else:
                own.extend(declared_here(c, src))
                walk(c)
        walk(scope)
        return own, kids

    def visit(scope, outer):
        own, kids = collect(scope)
        seen = {}
        for name, line in own:
            if name in seen:
                out.append((line, 'CS0128 同一層重複宣告', f'{name}（第 {seen[name]} 行也有）'))
            elif name in outer:
                out.append((line, 'CS0136 和外層的區域變數同名', f'{name}（外層在第 {outer[name]} 行）'))
            seen.setdefault(name, line)
        merged = dict(outer)
        merged.update(seen)
        for k in kids:
            visit(k, merged)

    def methods(n):
        for c in n.children:
            if c.type in ('method_declaration', 'constructor_declaration', 'accessor_declaration'):
                visit(c, {})
            else:
                methods(c)
    methods(tree.root_node)
    return out


def calls(tree, src):
    decl, members, public, local = {}, {}, {}, set()

    def gather(n, cls=None):
        if n.type == 'class_declaration':
            cls = text(n.child_by_field_name('name'), src)
            members.setdefault(cls, set())
            public.setdefault(cls, set())
        mods = ' '.join(text(c, src) for c in n.children if c.type == 'modifier')
        names = []
        if n.type in ('method_declaration', 'local_function_statement'):
            name = text(n.child_by_field_name('name'), src)
            ps = n.child_by_field_name('parameters')
            params = [c for c in ps.children if c.type == 'parameter'] if ps is not None else []
            many = any(text(c, src).startswith('params ') for c in params)
            need = sum(1 for c in params if not any(k.type == '=' for k in c.children) and not text(c, src).startswith('params '))
            decl.setdefault(name, []).append((need, 10 ** 6 if many else len(params)))
            names.append(name)
        elif n.type == 'field_declaration':
            for c in n.children:
                if c.type == 'variable_declaration':
                    names += [text(d.children[0], src) for d in c.children if d.type == 'variable_declarator']
        elif n.type in ('property_declaration', 'enum_declaration') or (n.type == 'class_declaration' and n.parent.type == 'declaration_list'):
            names.append(text(n.child_by_field_name('name'), src))
        owner = cls
        if n.type == 'class_declaration':                       # 巢狀類別是外層類別的成員
            p = n.parent
            while p is not None and p.type != 'class_declaration':
                p = p.parent
            owner = text(p.child_by_field_name('name'), src) if p is not None else None
        if owner and n.type != 'local_function_statement':
            for name in names:
                members[owner].add(name)
                if 'public' in mods or 'internal' in mods:
                    public[owner].add(name)
        for name, _ in declared_here(n, src):
            local.add(name)
        if n.type == 'local_function_statement':
            local.add(text(n.child_by_field_name('name'), src))
        for c in n.children:
            gather(c, cls)
    gather(tree.root_node)

    out = []

    def check(n, cls=None):
        if n.type == 'class_declaration':
            cls = text(n.child_by_field_name('name'), src)
        if n.type == 'invocation_expression':
            fn, args = n.child_by_field_name('function'), n.child_by_field_name('arguments')
            count = sum(1 for c in args.children if c.type == 'argument') if args is not None else 0
            name = None
            if fn.type == 'identifier':
                name = text(fn, src)
                if name not in decl and name not in local and name not in INHERITED and name not in ('nameof',):
                    out.append((n.start_point[0] + 1, '沒有宣告的方法', name))
            elif fn.type == 'member_access_expression' and text(fn.child_by_field_name('expression'), src) in members:
                name = text(fn.child_by_field_name('name'), src)
            if name in decl and not any(lo <= count <= hi for lo, hi in decl[name]):
                out.append((n.start_point[0] + 1, '參數個數不對', f'{name}：給了 {count} 個，宣告是 {decl[name]}'))
        if n.type == 'member_access_expression':
            ex = n.child_by_field_name('expression')
            if ex.type == 'identifier' and text(ex, src) in members:
                owner, name = text(ex, src), text(n.child_by_field_name('name'), src)
                if name not in members[owner]:
                    out.append((n.start_point[0] + 1, '沒有宣告的成員', f'{owner}.{name}'))
                elif owner != cls and name not in public[owner]:
                    out.append((n.start_point[0] + 1, '不是 public（CS0122）', f'{owner}.{name}'))
        for c in n.children:
            check(c, cls)
    check(tree.root_node)
    return out


def check_source(p, src):
    tree = p.parse(src)
    return syntax(tree, src) + scopes(tree, src) + calls(tree, src)


def test_checker_catches_known_mistakes(p):
    """檢查器本身：故意寫錯的都要抓得到，寫對的不能誤報。"""
    bad = b'''class A { public static int N; static int H(int a, int b = 0) { return a; }
        void F(int q) { int x = 1; { int x = 2; } System.Func<int,int> f = z => { int q = 3; return z; }; int w; int w;
        if (x > 0) { float lo = 1; } float lo = 2; H(); G(1); }
        void K() { if (true) { } int y = 0; else if (false) { } } }
        class B { static int P; } class C { void M() { int v = A.Nope + B.P; } }'''
    kinds = sorted({k for _, k, _ in check_source(p, bad)})
    assert kinds == ['CS0128 同一層重複宣告', 'CS0136 和外層的區域變數同名', 'syntax', '不是 public（CS0122）', '參數個數不對', '沒有宣告的成員', '沒有宣告的方法'], kinds
    good = b'''class A { public static int N; static int H(int a, int b = 0) { return a; }
        void F(int[] xs) { foreach (var p in System.Linq.Enumerable.Where(xs, p => p > 0)) { var y = p; } foreach (var p in xs) { var y = p; }
        for (int i = 0; i < 2; i++) { } for (int i = 0; i < 2; i++) { } int n; if (int.TryParse("1", out n)) H(n); H(1, 2); }
        class In { public int V; } } class C { void M() { int v = A.N; } }'''
    assert check_source(p, good) == [], check_source(p, good)
    print('  檢查器：7 種錯誤都抓得到，正確的寫法沒有誤報')


def test_files(p):
    total = 0
    for f in FILES:
        with open(os.path.join(ROOT, f), 'rb') as fh:
            src = fh.read()
        problems = sorted(check_source(p, src))
        for line, kind, what in problems:
            print(f'  {f}:{line}: {kind}: {what}')
        total += len(problems)
        print(f'  {f}：{src.count(bytes([10]))} 行，{len(problems)} 個問題')
    assert total == 0


def main():
    p = parser()
    if p is None:
        if os.environ.get('CI'):
            raise SystemExit('CI 上要有 tree-sitter 和 tree-sitter-c-sharp（pip install）')
        print('跳過：沒有安裝 tree-sitter／tree-sitter-c-sharp')
        return
    for f in (test_checker_catches_known_mistakes, test_files):
        print(f.__name__)
        f(p)
    print('OK')


if __name__ == '__main__':
    main()
