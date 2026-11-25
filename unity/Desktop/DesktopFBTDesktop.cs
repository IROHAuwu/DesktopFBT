// DesktopFBT — 桌面模式：讓 DesktopFBT 用 OSC 送來的參數直接驅動 Avatar 的姿勢（VRChat 桌面模式也能用）
// Tools > デスクトップFBT > Desktop Mode (デスクトップモード)
// 介面文字：日文（預設）／英文，在視窗最上面切換
#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;
using VRC.SDKBase;

namespace DesktopFBT
{
    // 介面文字：日文（預設）／英文。選擇記在這台電腦（EditorPrefs），所有專案共用
    public static class Loc
    {
        const string Key = "DesktopFBT.Lang";
        static int cached = -1;
        public static bool En
        {
            get
            {
                if (cached < 0) cached = EditorPrefs.GetString(Key, "ja") == "en" ? 1 : 0;
                return cached == 1;
            }
            set
            {
                cached = value ? 1 : 0;
                EditorPrefs.SetString(Key, value ? "en" : "ja");
            }
        }
        public static string T(string ja, string en) { return En ? en : ja; }
    }

    public static class DesktopRig
    {
        public const string Prefix = "YFBT/";
        public const string OnParam = Prefix + "On";
        public const string LayerName = "DesktopFBT Desktop";
        public const string SmoothLayer = "DesktopFBT Smooth";
        public const string HoldLayer = "DesktopFBT Hold Tracking";
        const float HoldEvery = 0.1f;      // 秒：Avatar 原本的跳躍／落地動畫會把追蹤改回來，開著時每隔這麼久重新設定一次
        const float SmoothA = 0.12f;       // 別人那邊每幀往新值靠近的比例
        public const string ObjectName = "DesktopFBT Desktop";
        public const string Folder = "Assets/DesktopFBT/Desktop";
        public const string MenuName = "デスクトップFBT";          // VRChat 選單上的名稱
        // 改名前（v1.7 以前）裝在 Avatar 上的名稱：移除／重新追加時也要認得
        const string Old = "Youtube" + "FBT";
        static readonly string OldObject = Old + " Desktop";
        static readonly string OldFolder = "Assets/" + Old + "/Desktop";
        static bool IsLayer(string name, string ours) { return name == ours || name == Old + ours.Substring("DesktopFBT".Length); }

        // 和 DesktopFBT 的 desktop.py 一致
        public static readonly int[] Muscles = {
            0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 };
        public static readonly int[] SyncLite = { 0, 1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35 };
        public static readonly string[] Body = { "YawS", "YawC", "TiltX", "TiltZ", "Y" };
        const float K = 10000f;            // 身體旋轉的權重：壓過其他子樹帶進來的單位旋轉

        public static string MuscleParam(int i) { return Prefix + "m" + i; }

        public static IEnumerable<string> FloatParams()
        {
            foreach (int m in Muscles) yield return MuscleParam(m);
            foreach (var b in Body) yield return Prefix + b;
        }

        public enum Sync { EveryoneBits, Everyone, LegsAndBody, LocalOnly }

        // EveryoneBits：每個值拆成幾個同步 Bool（高位在前，_0 = 最高位）。分配表有幾種，依 Avatar 剩下的 bit 選最大的一種；
        // DesktopFBT 每一種都會送（參數名稱裡有表的編號），不需要知道 Avatar 用的是哪一種。數字和 DesktopFBT 的 sync.py 一致。
        // 每一列的順序 = FloatParams()（44 個肌肉，然後 YawS、YawC、TiltX、TiltZ、Y）。0 = 這個值沒有自己的 bit。
        public static readonly int[][] Profiles = {
            new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 3, 0, 0, 0, 4, 3, 0, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 4, 3, 5, 4, 4 },
            new[] { 4, 3, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 5, 4, 0, 4, 0, 0, 0, 5, 4, 0, 4, 0, 0, 0, 0, 0, 4, 4, 4, 3, 0, 0, 0, 0, 0, 4, 4, 4, 3, 0, 0, 0, 4, 4, 5, 5, 6 },
            new[] { 4, 4, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 5, 5, 3, 4, 3, 2, 0, 5, 5, 3, 4, 3, 2, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 4, 4, 6, 5, 6 },
            new[] { 4, 5, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, 5, 6, 3, 5, 4, 3, 0, 5, 6, 3, 5, 4, 3, 0, 0, 0, 5, 5, 4, 5, 3, 2, 0, 0, 0, 5, 5, 4, 5, 3, 2, 0, 5, 5, 6, 6, 6 },
            new[] { 4, 5, 3, 0, 0, 0, 4, 3, 4, 0, 0, 0, 5, 6, 4, 6, 5, 3, 0, 5, 6, 4, 6, 5, 3, 0, 3, 2, 5, 6, 5, 5, 3, 4, 3, 3, 2, 5, 6, 5, 5, 3, 4, 3, 6, 6, 6, 6, 6 },
            new[] { 6, 6, 3, 0, 0, 0, 5, 5, 5, 0, 0, 0, 6, 6, 6, 6, 5, 4, 3, 6, 6, 6, 6, 5, 4, 3, 3, 2, 6, 6, 6, 6, 4, 5, 4, 3, 2, 6, 6, 6, 6, 4, 5, 4, 6, 6, 6, 6, 6 }
        };
        // 胸跟著脊椎、頭跟著脖子（兩根骨頭轉一樣多）：FollowFrom 的肌肉用 FollowTo 的 bit
        static readonly int[] FollowFrom = { 3, 4, 5, 12, 13, 14 };
        static readonly int[] FollowTo = { 0, 1, 2, 9, 10, 11 };
        public const int MaxBits = 6;
        public static int Profile = -1;
        public static Dictionary<string, int> Alloc = new Dictionary<string, int>();

        public static int Bits(string p) { int n; return Alloc.TryGetValue(p, out n) ? n : 0; }
        public static string BitParam(string p, int i) { return Prefix + "P" + Profile + "/" + p.Substring(Prefix.Length) + "_" + i; }

        // 這個值的同步 bit 從哪個參數來（跟著別人的肌肉 → 帶頭的那個）
        public static string BitSource(string p)
        {
            for (int i = 0; i < FollowFrom.Length; i++)
                if (p == MuscleParam(FollowFrom[i])) return MuscleParam(FollowTo[i]);
            return p;
        }

        // 同步的範圍：還原的值 = lo + Σ bit_k · step · 2^(n-1-k)，step = (hi - lo) / (格數 - 1)。
        // zero：有「0」這一格（格數 = 2^n - 1）。以前還原的是每一格的中間，0 表示不出來，站直時別人看到的身體是歪的。
        static void SyncRange(string p, out float lo, out float hi, out bool zero)
        {
            lo = -1f; hi = 1f; zero = true;
            if (p == Prefix + "TiltX" || p == Prefix + "TiltZ") { lo = -0.5f; hi = 0.5f; }
            else if (p == Prefix + "Y") { lo = -0.9f; hi = 0.5f; zero = false; }
        }

        public static int ProfileCost(int i) { return Profiles[i].Sum(); }

        public static int ProfileFor(int budget)
        {
            int best = -1;
            for (int i = 0; i < Profiles.Length; i++)
                if (ProfileCost(i) <= budget) best = i;
            return best;
        }

        // 第 k 種分配表（k < 0 = 全部是 0，只有自己看得到）
        public static Dictionary<string, int> AllocOf(int k)
        {
            var a = new Dictionary<string, int>();
            int j = 0;
            foreach (var p in FloatParams())
            {
                a[p] = k < 0 ? 0 : Profiles[k][j];
                j++;
            }
            return a;
        }

        // budget：給 DesktopFBT 用的 bit（不含開關）→ 放得下的最大那一種
        public static Dictionary<string, int> Allocate(int budget) { return AllocOf(ProfileFor(budget)); }

        public static int AllocCost(Dictionary<string, int> a) { return 1 + a.Values.Sum(); }

        // 上傳後其他的同步參數（名稱 → "Bool"／"Int"／"Float"）：Avatar 的 Expression Parameters（沒有設定時是 VRChat 預設的 3 個，24 bit），
        // 加上其他 MA Parameters。其他工具在上傳時才加的參數看不到，所以只是估計。
        static Dictionary<string, string> OtherParams(VRCAvatarDescriptor av)
        {
            var d = new Dictionary<string, string>();
            var ep = av.customExpressions ? av.expressionParameters : null;
            if (ep == null)
            {
                d["VRCEmote"] = "Int";
                d["VRCFaceBlendH"] = "Float";
                d["VRCFaceBlendV"] = "Float";
            }
            else if (ep.parameters != null)
            {
                foreach (var p in ep.parameters)
                    if (p != null && p.networkSynced && !string.IsNullOrEmpty(p.name) && !p.name.StartsWith(Prefix))
                        d[p.name] = p.valueType.ToString();
            }
            var t = FindType(MA + "ModularAvatarParameters");
            int unnamed = 0;
            if (t != null)
                foreach (var c in av.GetComponentsInChildren(t, true))
                {
                    if (c.transform.name == ObjectName || c.transform.name == OldObject) continue;
                    foreach (var cfg in (IEnumerable)t.GetField("parameters").GetValue(c))
                    {
                        var ty = cfg.GetType();
                        string st = ty.GetField("syncType").GetValue(cfg).ToString();
                        if ((bool)ty.GetField("localOnly").GetValue(cfg) || st == "NotSynced") continue;
                        var nameField = ty.GetField("nameOrPrefix");
                        var remapField = ty.GetField("remapTo");
                        string name = nameField != null ? nameField.GetValue(cfg) as string : null;
                        string remap = remapField != null ? remapField.GetValue(cfg) as string : null;
                        if (!string.IsNullOrEmpty(remap)) name = remap;
                        if (string.IsNullOrEmpty(name)) { name = "?" + unnamed; unnamed++; }
                        d[name] = st;
                    }
                }
            return d;
        }

        static int CostOf(Dictionary<string, string> others) { return others.Values.Sum(v => v == "Bool" ? 1 : 8); }
        public static int OtherCost(VRCAvatarDescriptor av) { return CostOf(OtherParams(av)); }

        // ------------------------------------------------------------ 其他同步參數交給 Param Compressor（同梱的或 lilMCR 的），空出來的 bit 拿來用
        // Compress：算空間時假設其他同步參數會被壓縮。Param Compressor 只在上傳時超過上限才壓，而且只壓到剛好放得下；
        // デスクトップFBT 自己的 Bool 會排除在外（壓縮的參數是輪流同步的，會慢）。
        public static bool Compress;
        public const int DefaultSteps = 8;
        // 最後一次 AllocFor 的結果（給介面顯示）：其他參數壓縮前／後的 bit、輪流的次數
        public static int OtherRaw, OtherPacked, PackSteps;
        // 重新追加時：原本 Param Compressor 上設定的輪流次數（我們的物件會先被刪掉再重建；> 0 時沿用）
        static int keepSteps;

        static int StepsOn(VRCAvatarDescriptor av)
        {
            var comp = FindCompressor(av);
            var f = comp == null ? null : comp.GetType().GetField("maxSyncSteps");
            return f != null ? Mathf.Clamp((int)f.GetValue(comp), 2, 32) : 0;
        }

        // Param Compressor 的算法（同梱的 DesktopFBTParamCompressorPass.MakePlan，和 lilMCR 的一樣）：numbers 個 Int／Float、bools 個 Bool 分 steps 次輪流時省幾 bit
        static int PlanSaved(int numbers, int bools, int steps, int maxSteps, out int states)
        {
            int nps = Mathf.Clamp(Mathf.CeilToInt((float)numbers / steps), 1, 8);
            int bps = Mathf.Clamp(Mathf.CeilToInt((float)bools / steps), 1, 64);
            bool useN = numbers > nps, useB = bools > bps;
            states = Math.Max(useN ? Mathf.CeilToInt((float)numbers / nps) : 0, useB ? Mathf.CeilToInt((float)bools / bps) : 0);
            if (states == 0 || states > maxSteps) return 0;
            int pointer = 1;
            if (states > 1)
            {
                pointer = 0;
                for (int v = states - 1; v > 0; v >>= 1) pointer++;
            }
            return (useN ? 8 * numbers - 8 * nps : 0) + (useB ? bools - bps : 0) - pointer;
        }

        // 要省下 need bit 時實際會省多少（和 ChoosePlan 一樣：次數少的先試，同一個次數先只壓 Bool、再只壓數值、再兩種都壓）。
        // 省不到 need 時回傳能省的最大值。states = 輪流的次數
        static int BestSaved(int numbers, int bools, int need, int maxSteps, out int states)
        {
            int best = 0;
            states = 0;
            for (int steps = 2; steps <= maxSteps; steps++)
                for (int variant = 0; variant < 3; variant++)
                {
                    int st;
                    int saved = PlanSaved(variant == 0 ? 0 : numbers, variant == 1 ? 0 : bools, steps, maxSteps, out st);
                    if (saved <= 0) continue;
                    if (saved >= need) { states = st; return saved; }
                    if (saved > best) { best = saved; states = st; }
                }
            return best;
        }

        // 其他參數裡可以被壓縮的有幾個（Param Compressor 自己的排除規則：VRChat 內建、臉部追蹤、使用者在 Param Compressor 上設定的排除）
        static void CountCompressible(VRCAvatarDescriptor av, Dictionary<string, string> others, out int numbers, out int bools, out int maxSteps)
        {
            numbers = 0;
            bools = 0;
            maxSteps = keepSteps > 0 ? keepSteps : DefaultSteps;
            if (!HasCompressor) return;
            var comp = FindCompressor(av);
            if (comp != null) maxSteps = StepsOn(av);
            var excluded = ExcludedMethod(comp);
            foreach (var kv in others)
            {
                var vt = kv.Value == "Bool" ? VRCExpressionParameters.ValueType.Bool
                       : kv.Value == "Int" ? VRCExpressionParameters.ValueType.Int : VRCExpressionParameters.ValueType.Float;
                bool skip = false;
                if (excluded != null)
                {
                    try { skip = (bool)excluded.Invoke(null, new object[] { kv.Key, vt, comp }); }
                    catch (Exception) { skip = false; }
                }
                if (skip) continue;
                if (kv.Value == "Bool") bools++;
                else numbers++;
            }
        }

        // Param Compressor 的「不壓縮的開頭」裡要不要有デスクトップFBT 的參數（Bool 同步時要排除；用 Param Compressor 同步全身的 Float 時不能排除）
        static void SetOurExclusion(Component comp, bool on)
        {
            var f = comp.GetType().GetField("excludedPrefixes");
            if (f == null) return;
            var list = ((string[])f.GetValue(comp) ?? new string[0]).Where(x => x != Prefix).ToList();
            if (on) list.Add(Prefix);
            Undo.RecordObject(comp, "DesktopFBT Desktop");
            f.SetValue(comp, list.ToArray());
            EditorUtility.SetDirty(comp);
        }

        // 用壓縮來空出 bit：Avatar 上沒有 Param Compressor 就加在 host 上（host 是 null 時自己建一個物件），並把デスクトップFBT 的 Bool 排除
        static void PrepareCompressor(VRCAvatarDescriptor av, GameObject host)
        {
            var comp = FindCompressor(av);
            if (comp == null)
            {
                var t = NewCompressorType();
                if (t == null) return;
                if (host == null)
                {
                    host = new GameObject(ObjectName);
                    Undo.RegisterCreatedObjectUndo(host, "DesktopFBT Desktop");
                    host.transform.SetParent(av.transform, false);
                }
                comp = host.AddComponent(t);
                SetField(comp, "maxSyncSteps", keepSteps > 0 ? keepSteps : DefaultSteps);
            }
            SetOurExclusion(comp, true);
        }

        // 同步的細かさ：-1 = 自動（空きに入る最大のもの）、0 以上 = Profiles の番号を使用者が指定
        public static int Level = -1;
        // 最後に AllocFor した Avatar で DesktopFBT が使える bit（オン／オフのスイッチを除く）
        public static int Budget;

        public static Dictionary<string, int> AllocFor(VRCAvatarDescriptor av)
        {
            var others = OtherParams(av);
            int limit = VRCExpressionParameters.MAX_PARAMETER_COST;
            OtherRaw = CostOf(others);
            OtherPacked = OtherRaw;
            PackSteps = 0;
            int most = 0, numbers = 0, bools = 0, maxSteps = DefaultSteps, st;
            if (Compress && HasCompressor)
            {
                CountCompressible(av, others, out numbers, out bools, out maxSteps);
                most = BestSaved(numbers, bools, int.MaxValue, maxSteps, out st);      // 最多能省多少 → 決定放得下哪一種
            }
            Budget = limit - 1 - (OtherRaw - most);
            Profile = Level < 0 ? ProfileFor(Budget) : Mathf.Min(Level, Profiles.Length - 1);
            int need = OtherRaw + 1 + (Profile < 0 ? 0 : ProfileCost(Profile)) - limit;  // 選了這一種之後實際要省多少
            if (most > 0 && need > 0)
            {
                OtherPacked = OtherRaw - BestSaved(numbers, bools, need, maxSteps, out st);
                PackSteps = st;
            }
            return AllocOf(Profile);
        }

        // 選んだ細かさが VRChat の上限を何 bit 超えるか（0 以下 = 収まる）。超えたままではアップロードできない。
        public static int Over() { return Profile < 0 ? 0 : ProfileCost(Profile) - Budget; }

        public static bool IsSynced(string name, Sync sync)
        {
            if (sync == Sync.EveryoneBits) return false;     // Float 只給自己用，同步的是 Bool
            if (sync == Sync.Everyone) return true;
            if (sync == Sync.LocalOnly) return false;
            foreach (var b in Body) if (name == Prefix + b) return true;
            foreach (int m in SyncLite) if (name == MuscleParam(m)) return true;
            return false;
        }

        // 姿勢用的是平滑後的值（別人那邊參數是一段一段更新的）
        public static string Smoothed(string p) { return Prefix + "S/" + p.Substring(Prefix.Length); }

        // ------------------------------------------------------------ 產生 Animator Controller
        static AnimationClip Clip(string name, Action<AnimationClip> fill, bool root)
        {
            var c = new AnimationClip { name = name };
            fill(c);
            var s = AnimationUtility.GetAnimationClipSettings(c);
            s.loopTime = true;
            if (root)
            {
                s.keepOriginalOrientation = true; s.loopBlendOrientation = true;
                s.keepOriginalPositionY = true; s.loopBlendPositionY = true;
                s.keepOriginalPositionXZ = true; s.loopBlendPositionXZ = true;
            }
            AnimationUtility.SetAnimationClipSettings(c, s);
            return c;
        }

        static void Curve(AnimationClip c, string prop, float v)
        {
            c.SetCurve("", typeof(Animator), prop, AnimationCurve.Constant(0f, 1f, v));
        }

        static void EnsureFolder()
        {
            if (!AssetDatabase.IsValidFolder("Assets/DesktopFBT")) AssetDatabase.CreateFolder("Assets", "DesktopFBT");
            if (!AssetDatabase.IsValidFolder(Folder)) AssetDatabase.CreateFolder("Assets/DesktopFBT", "Desktop");
        }

        public static AnimatorController Build(bool headFollow, bool bits)
        {
            EnsureFolder();
            string path = Folder + "/DesktopFBT_Desktop" + (headFollow ? "_Head" : "") + (bits ? "_Sync" + Profile : "") + ".controller";
            AssetDatabase.DeleteAsset(path);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            Populate(ctrl, headFollow, bits);
            ctrl.RemoveLayer(0);
            AssetDatabase.SaveAssets();
            return ctrl;
        }

        // 參數、圖層與動畫都加進 ctrl（子資產名稱都以 "YFBT " 開頭，移除時用來辨認）
        public static void Populate(AnimatorController ctrl, bool headFollow, bool bits)
        {
            var assets = new List<UnityEngine.Object>();
            var defaults = new Dictionary<string, float>();
            Action<string, AnimatorControllerParameterType, float> param = (n, t, d) =>
            {
                if (ctrl.parameters.Any(x => x.name == n)) return;
                ctrl.AddParameter(n, t);
                if (d != 0f) defaults[n] = d;
            };
            param(OnParam, AnimatorControllerParameterType.Bool, 0f);
            foreach (var p in FloatParams()) param(p, AnimatorControllerParameterType.Float, 0f);
            foreach (var p in FloatParams()) param(Smoothed(p), AnimatorControllerParameterType.Float, 0f);
            if (bits)
                foreach (var p in FloatParams())
                    for (int i = 0; i < Bits(p); i++) param(BitParam(p, i), AnimatorControllerParameterType.Float, 0f);
            param(Prefix + "One", AnimatorControllerParameterType.Float, 1f);
            param(Prefix + "K", AnimatorControllerParameterType.Float, K);
            param(Prefix + "A", AnimatorControllerParameterType.Float, SmoothA);
            param(Prefix + "B", AnimatorControllerParameterType.Float, 1f - SmoothA);
            param("IsLocal", AnimatorControllerParameterType.Bool, 0f);
            var all = ctrl.parameters;
            foreach (var p in all) if (defaults.ContainsKey(p.name)) p.defaultFloat = defaults[p.name];
            ctrl.parameters = all;

            var root = new BlendTree { name = "YFBT Pose", blendType = BlendTreeType.Direct, hideFlags = HideFlags.HideInHierarchy };
            assets.Add(root);
            var kids = new List<ChildMotion>();

            foreach (int m in Muscles)
            {
                string mn = HumanTrait.MuscleName[m];
                var lo = Clip("YFBT " + mn + " -1", c => Curve(c, mn, -1f), false);
                var hi = Clip("YFBT " + mn + " +1", c => Curve(c, mn, 1f), false);
                var t = new BlendTree { name = "YFBT " + mn, blendType = BlendTreeType.Simple1D, blendParameter = Smoothed(MuscleParam(m)),
                                        useAutomaticThresholds = false, hideFlags = HideFlags.HideInHierarchy };
                t.AddChild(lo, -1f);
                t.AddChild(hi, 1f);
                assets.Add(lo); assets.Add(hi); assets.Add(t);
                kids.Add(new ChildMotion { motion = t, directBlendParameter = Prefix + "One", timeScale = 1f });
            }

            // 身體高度
            {
                var lo = Clip("YFBT Body Y 0", c => { Curve(c, "RootT.x", 0f); Curve(c, "RootT.y", 0f); Curve(c, "RootT.z", 0f); }, true);
                var hi = Clip("YFBT Body Y 1.5", c => { Curve(c, "RootT.x", 0f); Curve(c, "RootT.y", 1.5f); Curve(c, "RootT.z", 0f); }, true);
                var t = new BlendTree { name = "YFBT Body Y", blendType = BlendTreeType.Simple1D, blendParameter = Smoothed(Prefix + "Y"),
                                        useAutomaticThresholds = false, hideFlags = HideFlags.HideInHierarchy };
                t.AddChild(lo, -1f);
                t.AddChild(hi, 1f);
                assets.Add(lo); assets.Add(hi); assets.Add(t);
                kids.Add(new ChildMotion { motion = t, directBlendParameter = Prefix + "One", timeScale = 1f });
            }

            // 身體旋轉：偏航（方向）→ 傾斜（2D）的格點。q = Euler(0, 偏航, 0) · 繞水平軸 (x, 0, z) 轉 |(x, z)|·180°
            // （和 desktop.py 的 body_tilt 一樣；尤拉角在彎腰 90° 時會萬向鎖，這種拆法除了倒立以外都連續）
            {
                var yawTree = new BlendTree { name = "YFBT Body Rotation", blendType = BlendTreeType.FreeformDirectional2D,
                                              blendParameter = Smoothed(Prefix + "YawS"), blendParameterY = Smoothed(Prefix + "YawC"),
                                              hideFlags = HideFlags.HideInHierarchy };
                assets.Add(yawTree);
                for (int yaw = -135; yaw <= 180; yaw += 45)
                {
                    var g = new BlendTree { name = "YFBT Yaw " + yaw, blendType = BlendTreeType.FreeformCartesian2D,
                                            blendParameter = Smoothed(Prefix + "TiltX"), blendParameterY = Smoothed(Prefix + "TiltZ"),
                                            hideFlags = HideFlags.HideInHierarchy };
                    assets.Add(g);
                    for (int ix = -5; ix <= 5; ix++)
                        for (int iz = -5; iz <= 5; iz++)
                        {
                            var t = new Vector2(ix / 6f, iz / 6f);
                            var q = Quaternion.Euler(0f, yaw, 0f);
                            if (t.magnitude > 1e-6f)
                                q = q * Quaternion.AngleAxis(t.magnitude * 180f, new Vector3(t.x, 0f, t.y) / t.magnitude);
                            var clip = Clip("YFBT Body " + yaw + " " + ix + " " + iz, c =>
                            {
                                Curve(c, "RootQ.x", q.x); Curve(c, "RootQ.y", q.y); Curve(c, "RootQ.z", q.z); Curve(c, "RootQ.w", q.w);
                            }, true);
                            assets.Add(clip);
                            g.AddChild(clip, t);
                        }
                    float r = yaw * Mathf.Deg2Rad;
                    yawTree.AddChild(g, new Vector2(Mathf.Sin(r), Mathf.Cos(r)));
                }
                kids.Add(new ChildMotion { motion = yawTree, directBlendParameter = Prefix + "K", timeScale = 1f });
            }
            root.children = kids.ToArray();

            // 平滑：自己直接用收到的值；別人那邊每幀往收到的值靠近（Animator 參數動畫）
            var copy = new BlendTree { name = "YFBT Copy", blendType = BlendTreeType.Direct, hideFlags = HideFlags.HideInHierarchy };
            var smooth = new BlendTree { name = "YFBT Smooth", blendType = BlendTreeType.Direct, hideFlags = HideFlags.HideInHierarchy };
            assets.Add(copy); assets.Add(smooth);
            var ck = new List<ChildMotion>();
            var sk = new List<ChildMotion>();
            foreach (var p in FloatParams())
            {
                string sp = Smoothed(p);
                var lo = new AnimationClip { name = "YFBT " + sp + " -1" };
                lo.SetCurve("", typeof(Animator), sp, AnimationCurve.Constant(0f, 1f, -1f));
                var hi = new AnimationClip { name = "YFBT " + sp + " +1" };
                hi.SetCurve("", typeof(Animator), sp, AnimationCurve.Constant(0f, 1f, 1f));
                assets.Add(lo); assets.Add(hi);
                Func<string, BlendTree> drive = src =>
                {
                    var t = new BlendTree { name = "YFBT " + sp + " <- " + src, blendType = BlendTreeType.Simple1D, blendParameter = src,
                                            useAutomaticThresholds = false, hideFlags = HideFlags.HideInHierarchy };
                    t.AddChild(lo, -1f);
                    t.AddChild(hi, 1f);
                    assets.Add(t);
                    return t;
                };
                ck.Add(new ChildMotion { motion = drive(p), directBlendParameter = Prefix + "One", timeScale = 1f });
                Motion target = drive(p);
                string bp = BitSource(p);
                if (bits && Bits(bp) > 0)
                {
                    // 收到的 Bool → 數值：lo + Σ bit_k · step · 2^(n-1-k)（k = 0 是最高位）
                    int n = Bits(bp);
                    float vmin, vmax; bool zero;
                    SyncRange(bp, out vmin, out vmax, out zero);
                    float step = (vmax - vmin) / ((1 << n) - (zero ? 2 : 1));
                    var dec = new BlendTree { name = "YFBT " + sp + " bits", blendType = BlendTreeType.Direct, hideFlags = HideFlags.HideInHierarchy };
                    var bias = new AnimationClip { name = "YFBT " + sp + " bias" + n };
                    bias.SetCurve("", typeof(Animator), sp, AnimationCurve.Constant(0f, 1f, vmin));
                    assets.Add(bias);
                    var dk = new List<ChildMotion> { new ChildMotion { motion = bias, directBlendParameter = Prefix + "One", timeScale = 1f } };
                    for (int i = 0; i < n; i++)
                    {
                        var c = new AnimationClip { name = "YFBT " + sp + " bit" + i };
                        c.SetCurve("", typeof(Animator), sp, AnimationCurve.Constant(0f, 1f, step * (1 << (n - 1 - i))));
                        assets.Add(c);
                        dk.Add(new ChildMotion { motion = c, directBlendParameter = BitParam(bp, i), timeScale = 1f });
                    }
                    dec.children = dk.ToArray();
                    assets.Add(dec);
                    target = dec;
                }
                sk.Add(new ChildMotion { motion = target, directBlendParameter = Prefix + "A", timeScale = 1f });
                sk.Add(new ChildMotion { motion = drive(sp), directBlendParameter = Prefix + "B", timeScale = 1f });
            }
            copy.children = ck.ToArray();
            smooth.children = sk.ToArray();

            var empty = new AnimationClip { name = "YFBT Empty" };
            assets.Add(empty);
            foreach (var a in assets) AssetDatabase.AddObjectToAsset(a, ctrl);

            ctrl.AddLayer(SmoothLayer);
            {
                var ls = ctrl.layers;
                var l = ls[ls.Length - 1];
                l.defaultWeight = 1f;
                var m = l.stateMachine;
                var local = m.AddState("Local", new Vector3(300, 0, 0));
                local.motion = copy; local.writeDefaultValues = true;
                var remote = m.AddState("Remote", new Vector3(300, 120, 0));
                remote.motion = smooth; remote.writeDefaultValues = true;
                m.defaultState = remote;
                var tr = remote.AddTransition(local);
                tr.hasExitTime = false; tr.duration = 0f;
                tr.AddCondition(AnimatorConditionMode.If, 0, "IsLocal");
                var dr = remote.AddStateMachineBehaviour<VRCAvatarParameterDriver>();
                dr.localOnly = false;
                dr.parameters = new List<VRC_AvatarParameterDriver.Parameter> {
                    new VRC_AvatarParameterDriver.Parameter { type = VRC_AvatarParameterDriver.ChangeType.Set, name = Prefix + "One", value = 1f },
                    new VRC_AvatarParameterDriver.Parameter { type = VRC_AvatarParameterDriver.ChangeType.Set, name = Prefix + "A", value = SmoothA },
                    new VRC_AvatarParameterDriver.Parameter { type = VRC_AvatarParameterDriver.ChangeType.Set, name = Prefix + "B", value = 1f - SmoothA },
                };
                ctrl.layers = ls;
            }

            ctrl.AddLayer(LayerName);
            var layers = ctrl.layers;
            var layer = layers[layers.Length - 1];
            layer.defaultWeight = 1f;
            var sm = layer.stateMachine;
            var off = sm.AddState("Off", new Vector3(300, 0, 0));
            off.motion = empty;
            off.writeDefaultValues = true;
            var on = sm.AddState("On", new Vector3(300, 120, 0));
            on.motion = root;
            on.writeDefaultValues = true;
            sm.defaultState = off;

            var t1 = off.AddTransition(on);
            t1.hasExitTime = false; t1.duration = 0f;
            t1.AddCondition(AnimatorConditionMode.If, 0, OnParam);
            var t2 = on.AddTransition(off);
            t2.hasExitTime = false; t2.duration = 0f;
            t2.AddCondition(AnimatorConditionMode.IfNot, 0, OnParam);

            var drv = on.AddStateMachineBehaviour<VRCAvatarParameterDriver>();
            drv.localOnly = false;
            drv.parameters = new List<VRC_AvatarParameterDriver.Parameter> {
                new VRC_AvatarParameterDriver.Parameter { type = VRC_AvatarParameterDriver.ChangeType.Set, name = Prefix + "One", value = 1f },
                new VRC_AvatarParameterDriver.Parameter { type = VRC_AvatarParameterDriver.ChangeType.Set, name = Prefix + "K", value = K },
            };
            var plOn = on.AddStateMachineBehaviour<VRCPlayableLayerControl>();
            plOn.layer = VRC_PlayableLayerControl.BlendableLayer.Action;
            plOn.goalWeight = 1f;
            plOn.blendDuration = 0.2f;
            var tcOn = on.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();
            Tracking(tcOn, VRC_AnimatorTrackingControl.TrackingType.Animation, headFollow);

            var plOff = off.AddStateMachineBehaviour<VRCPlayableLayerControl>();
            plOff.layer = VRC_PlayableLayerControl.BlendableLayer.Action;
            plOff.goalWeight = 0f;
            plOff.blendDuration = 0.2f;
            var tcOff = off.AddStateMachineBehaviour<VRCAnimatorTrackingControl>();
            Tracking(tcOff, VRC_AnimatorTrackingControl.TrackingType.Tracking, headFollow);

            ctrl.layers = layers;

            // 一直維持「動畫控制」：很多 Avatar 的 Locomotion 在跳躍、落地時會把頭、身體改回玩家控制
            var tick = new AnimationClip { name = "YFBT Tick" };
            tick.SetCurve("__yfbt_tick", typeof(GameObject), "m_IsActive", AnimationCurve.Constant(0f, HoldEvery, 1f));
            AssetDatabase.AddObjectToAsset(tick, ctrl);
            ctrl.AddLayer(HoldLayer);
            var hl = ctrl.layers;
            var h = hl[hl.Length - 1];
            h.defaultWeight = 1f;
            var hm = h.stateMachine;
            var hOff = hm.AddState("Off", new Vector3(300, 0, 0));
            hOff.motion = empty; hOff.writeDefaultValues = true;
            var hA = hm.AddState("Hold A", new Vector3(300, 120, 0));
            var hB = hm.AddState("Hold B", new Vector3(560, 120, 0));
            hm.defaultState = hOff;
            var go = hOff.AddTransition(hA);
            go.hasExitTime = false; go.duration = 0f;
            go.AddCondition(AnimatorConditionMode.If, 0, OnParam);
            foreach (var pair in new[] { new[] { hA, hB }, new[] { hB, hA } })
            {
                var st = pair[0];
                st.motion = tick; st.writeDefaultValues = true;
                Tracking(st.AddStateMachineBehaviour<VRCAnimatorTrackingControl>(), VRC_AnimatorTrackingControl.TrackingType.Animation, headFollow);
                var stop = st.AddTransition(hOff);
                stop.hasExitTime = false; stop.duration = 0f;
                stop.AddCondition(AnimatorConditionMode.IfNot, 0, OnParam);
                var next = st.AddTransition(pair[1]);
                next.hasExitTime = true; next.exitTime = 1f; next.duration = 0f;
                next.AddCondition(AnimatorConditionMode.If, 0, OnParam);
            }
            ctrl.layers = hl;
            EditorUtility.SetDirty(ctrl);
        }

        static void Tracking(VRCAnimatorTrackingControl t, VRC_AnimatorTrackingControl.TrackingType v, bool head)
        {
            var keep = VRC_AnimatorTrackingControl.TrackingType.NoChange;
            t.trackingHead = head ? v : keep;
            t.trackingLeftHand = v;
            t.trackingRightHand = v;
            t.trackingHip = v;
            t.trackingLeftFoot = v;
            t.trackingRightFoot = v;
            t.trackingLeftFingers = keep;
            t.trackingRightFingers = keep;
            t.trackingEyes = keep;
            t.trackingMouth = keep;
        }

        // ------------------------------------------------------------ Modular Avatar（不破壞原本的 Avatar）
        static Type FindType(string name)
        {
            foreach (var a in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = a.GetType(name);
                if (t != null) return t;
            }
            return null;
        }

        const string MA = "nadena.dev.modular_avatar.core.";
        public static bool HasMA { get { return FindType(MA + "ModularAvatarMergeAnimator") != null; } }

        static void SetField(object o, string name, object v)
        {
            var f = o.GetType().GetField(name);
            if (f == null) throw new MissingFieldException(o.GetType().Name, name);
            if (f.FieldType.IsEnum && !(v != null && v.GetType().IsEnum)) v = Enum.Parse(f.FieldType, v.ToString());
            f.SetValue(o, v);
        }

        // ------------------------------------------------------------ Param Compressor（把同步參數輪流送）
        // 同梱的（unity/ParamCompressor，從 lilMCR Param Compressor 搬過來，沒裝 lilMCR 也能用）和 lilMCR 的都認得。
        // 兩個都是 NDMF 的 pass：沒有 NDMF（Modular Avatar）時 pass 不會被編譯，就當成沒有。
        const string OwnCompressorType = "DesktopFBT.DesktopFBTParamCompressor";
        const string OwnPassType = "DesktopFBT.DesktopFBTParamCompressorPass";
        const string CompressorType = "liltoon_MCR.MacaronParamCompressor";
        const string LilPassType = "liltoon_MCR.MacaronParamCompressorPass";
        static bool OwnReady { get { return FindType(OwnCompressorType) != null && FindType(OwnPassType) != null; } }
        static bool LilReady { get { return FindType(CompressorType) != null && FindType(LilPassType) != null; } }
        public static bool HasCompressor { get { return OwnReady || LilReady; } }

        // Avatar 上已經有的 Param Compressor（lilMCR 的優先，沒有再找同梱的）
        static Component FindCompressor(VRCAvatarDescriptor av)
        {
            foreach (var name in new[] { CompressorType, OwnCompressorType })
            {
                var t = FindType(name);
                var c = t == null ? null : av.GetComponentInChildren(t, true);
                if (c != null) return c;
            }
            return null;
        }

        // 要新加一個時用哪一種：同梱的優先（不依賴 lilMCR），沒有才用 lilMCR 的
        static Type NewCompressorType()
        {
            if (OwnReady) return FindType(OwnCompressorType);
            return LilReady ? FindType(CompressorType) : null;
        }

        // 那個 Param Compressor 判斷「這個參數不壓縮」的方法（comp 是 null 時用之後會加的那一種）
        static System.Reflection.MethodInfo ExcludedMethod(Component comp)
        {
            Type ct = comp != null ? comp.GetType() : NewCompressorType();
            if (ct == null) return null;
            var pass = FindType(ct.FullName == CompressorType ? LilPassType : OwnPassType);
            return pass == null ? null : pass.GetMethod("IsExcluded",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
        }

        public static string NeedCompressor
        {
            get
            {
                return Loc.T("パラメーターの圧縮には Modular Avatar（NDMF）が必要です。先に Modular Avatar をプロジェクトに入れてください。",
                             "Parameter compression needs Modular Avatar (NDMF). Add Modular Avatar to the project first.");
            }
        }

        // 回傳提示（沒有問題時為 null）
        static string EnsureCompressor(VRCAvatarDescriptor av, GameObject host)
        {
            var have = FindCompressor(av);
            if (have == null)
            {
                var c = host.AddComponent(NewCompressorType());
                SetField(c, "maxSyncSteps", 16);
                return null;
            }
            SetOurExclusion(have, false);
            var f = have.GetType().GetField("excludeFloats");
            if (f != null && (bool)f.GetValue(have))
                return Loc.T("アバターの Param Compressor で「Float を圧縮しない」がオンになっています。デスクトップFBT のパラメーターが上限を超えるので、オフにしてください。",
                             "The avatar's Param Compressor is set to leave Floats uncompressed. デスクトップFBT's parameters would exceed the limit; turn that option off.");
            return null;
        }

        public static GameObject AddWithMA(VRCAvatarDescriptor av, Sync sync, bool headFollow)
        {
            if (sync == Sync.Everyone && !HasCompressor) throw new InvalidOperationException(NeedCompressor);
            keepSteps = StepsOn(av);
            Remove(av);
            bool bits = sync == Sync.EveryoneBits;
            if (bits) Alloc = AllocFor(av);
            var ctrl = Build(headFollow, bits);
            var go = new GameObject(ObjectName);
            Undo.RegisterCreatedObjectUndo(go, "DesktopFBT Desktop");
            go.transform.SetParent(av.transform, false);

            var merge = go.AddComponent(FindType(MA + "ModularAvatarMergeAnimator"));
            SetField(merge, "animator", ctrl);
            SetField(merge, "layerType", "Action");
            SetField(merge, "pathMode", "Absolute");
            SetField(merge, "matchAvatarWriteDefaults", false);
            SetField(merge, "deleteAttachedAnimator", true);

            var pc = go.AddComponent(FindType(MA + "ModularAvatarParameters"));
            var cfgType = FindType(MA + "ParameterConfig");
            var list = (IList)pc.GetType().GetField("parameters").GetValue(pc);
            Action<string, string, bool, float> add = (n, type, synced, def) =>
            {
                object c = Activator.CreateInstance(cfgType);
                SetField(c, "nameOrPrefix", n);
                SetField(c, "syncType", type);
                SetField(c, "localOnly", !synced);
                SetField(c, "defaultValue", def);
                SetField(c, "saved", false);
                list.Add(c);
            };
            add(OnParam, "Bool", true, 0f);
            foreach (var p in FloatParams()) add(p, "Float", IsSynced(p, sync), 0f);
            if (bits)
                foreach (var p in FloatParams())
                    for (int i = 0; i < Bits(p); i++) add(BitParam(p, i), "Bool", true, 0f);
            add(Prefix + "One", "NotSynced", false, 1f);
            add(Prefix + "K", "NotSynced", false, K);

            go.AddComponent(FindType(MA + "ModularAvatarMenuInstaller"));
            var item = go.AddComponent(FindType(MA + "ModularAvatarMenuItem"));
            var ctl = new VRCExpressionsMenu.Control
            {
                name = MenuName,
                type = VRCExpressionsMenu.Control.ControlType.Toggle,
                parameter = new VRCExpressionsMenu.Control.Parameter { name = OnParam },
                value = 1f,
            };
            SetField(item, "Control", ctl);
            SetField(item, "label", MenuName);
            SetField(item, "isSynced", true);
            SetField(item, "isSaved", false);
            if (sync == Sync.Everyone) LastWarning = EnsureCompressor(av, go);
            if (bits) LastWarning = AllocWarning();
            if (bits && Compress && HasCompressor) PrepareCompressor(av, go);
            keepSteps = 0;
            EditorUtility.SetDirty(go);
            return go;
        }

        // ------------------------------------------------------------ 直接套用（沒有 Modular Avatar 時）
        public static string LastWarning;

        static string AllocWarning()
        {
            int need = ProfileCost(0) + 1, max = ProfileCost(Profiles.Length - 1) + 1;
            if (Over() > 0)
                return Loc.T("同期パラメーターが VRChat の上限（256 bit）を約 " + Over() + " bit 超えています。このままではアップロードできません。他の同期パラメーターを減らすか、「同期の細かさ」を下げてください。",
                             "Synced parameters exceed VRChat's limit (256 bits) by about " + Over() + " bits. The avatar cannot be uploaded like this. Reduce other synced parameters or lower \"Sync detail\".");
            if (Level >= 0) return null;        // 自分で選んだ細かさ：収まっていれば何も言わない
            if (Profile < 0)
                return Loc.T("同期パラメーターの空きが足りません（" + need + " bit 必要）。動きは自分にしか見えません。他の同期パラメーターを減らすか、「他の同期パラメーターを圧縮して空きを増やす」をオンにしてください。",
                             "Not enough free synced parameter space (" + need + " bits needed). The motion is visible only to you. Reduce other synced parameters or turn on \"Compress other synced parameters\".");
            if (Profile < Profiles.Length - 1)
                return Loc.T("同期に使える bit が少ないため、他の人には少し粗く見えます（" + AllocCost(Alloc) + " bit 使用、最大 " + max + " bit）。他の同期パラメーターを減らすと細かくなります。",
                             "Few bits are free for syncing, so other people see a slightly coarser motion (" + AllocCost(Alloc) + " bits used, up to " + max + "). Reducing other synced parameters makes it finer.");
            return null;
        }

        public static string Summary(Dictionary<string, int> a)
        {
            int synced = a.Keys.Count(k => { int n; return a.TryGetValue(BitSource(k), out n) && n > 0; });
            if (a.Values.Sum() == 0) return Loc.T("自分のみ（空きが足りません）", "local only (not enough free space)");
            return Loc.T(a.Count + " 個中 " + synced + " 個を同期、合計 ", synced + " of " + a.Count + " synced, total ") + AllocCost(a) + " bit";
        }

        public static string ApplyDirect(VRCAvatarDescriptor av, Sync sync, bool headFollow)
        {
            if (sync == Sync.Everyone && !HasCompressor) return NeedCompressor;
            keepSteps = StepsOn(av);
            Remove(av);
            if (sync == Sync.EveryoneBits) Alloc = AllocFor(av);
            EnsureFolder();
            Undo.RecordObject(av, "DesktopFBT Desktop");
            av.customizeAnimationLayers = true;
            var layers = av.baseAnimationLayers;
            if (layers == null || layers.Length == 0)
                layers = new[] { VRCAvatarDescriptor.AnimLayerType.Base, VRCAvatarDescriptor.AnimLayerType.Additive,
                                 VRCAvatarDescriptor.AnimLayerType.Gesture, VRCAvatarDescriptor.AnimLayerType.Action,
                                 VRCAvatarDescriptor.AnimLayerType.FX }
                    .Select(t => new VRCAvatarDescriptor.CustomAnimLayer { type = t, isDefault = true, isEnabled = t != VRCAvatarDescriptor.AnimLayerType.Additive })
                    .ToArray();
            int ai = Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.Action);
            if (ai < 0) return Loc.T("Action レイヤーが見つかりません。", "The Action layer was not found.");
            var target = layers[ai].animatorController as AnimatorController;
            if (layers[ai].isDefault || target == null)
            {
                layers[ai].isDefault = false;
                layers[ai].isEnabled = true;
                layers[ai].animatorController = DefaultActionCopy(headFollow, sync == Sync.EveryoneBits);
                av.baseAnimationLayers = layers;
            }
            else
            {
                Undo.RegisterCompleteObjectUndo(target, "DesktopFBT Desktop");
                Populate(target, headFollow, sync == Sync.EveryoneBits);
                EditorUtility.SetDirty(target);
            }

            var ep = av.expressionParameters;
            if (ep == null)
            {
                ep = ScriptableObject.CreateInstance<VRCExpressionParameters>();
                ep.parameters = new VRCExpressionParameters.Parameter[0];
                AssetDatabase.CreateAsset(ep, Folder + "/DesktopFBT_Parameters.asset");
                av.customExpressions = true;
                av.expressionParameters = ep;
            }
            Undo.RecordObject(ep, "DesktopFBT Desktop");
            var plist = ep.parameters.ToList();
            plist.Add(new VRCExpressionParameters.Parameter { name = OnParam, valueType = VRCExpressionParameters.ValueType.Bool, networkSynced = true });
            foreach (var p in FloatParams())
                plist.Add(new VRCExpressionParameters.Parameter { name = p, valueType = VRCExpressionParameters.ValueType.Float, networkSynced = IsSynced(p, sync) });
            if (sync == Sync.EveryoneBits)
                foreach (var p in FloatParams())
                    for (int i = 0; i < Bits(p); i++)
                        plist.Add(new VRCExpressionParameters.Parameter { name = BitParam(p, i), valueType = VRCExpressionParameters.ValueType.Bool, networkSynced = true });
            ep.parameters = plist.ToArray();
            EditorUtility.SetDirty(ep);
            string warn = null;
            if (sync == Sync.Everyone)
            {
                var host = new GameObject(ObjectName);
                Undo.RegisterCreatedObjectUndo(host, "DesktopFBT Desktop");
                host.transform.SetParent(av.transform, false);
                warn = EnsureCompressor(av, host);
            }
            else if (sync == Sync.EveryoneBits)
            {
                warn = AllocWarning();
                if (Compress && HasCompressor) PrepareCompressor(av, null);
            }
            else if (ep.CalcTotalCost() > VRCExpressionParameters.MAX_PARAMETER_COST)
                warn = Loc.T("同期パラメーターが上限を超えています。「自分のみ」に変更してください。",
                             "Synced parameters exceed the limit. Switch to \"Only me\".");
            keepSteps = 0;

            var menu = av.expressionsMenu;
            if (menu == null)
            {
                menu = ScriptableObject.CreateInstance<VRCExpressionsMenu>();
                AssetDatabase.CreateAsset(menu, Folder + "/DesktopFBT_Menu.asset");
                av.customExpressions = true;
                av.expressionsMenu = menu;
            }
            if (menu.controls.Count < VRCExpressionsMenu.MAX_CONTROLS)
            {
                Undo.RecordObject(menu, "DesktopFBT Desktop");
                menu.controls.Add(new VRCExpressionsMenu.Control
                {
                    name = MenuName,
                    type = VRCExpressionsMenu.Control.ControlType.Toggle,
                    parameter = new VRCExpressionsMenu.Control.Parameter { name = OnParam },
                    value = 1f,
                });
                EditorUtility.SetDirty(menu);
            }
            else warn = (warn == null ? "" : warn + "\n") + Loc.T("メニューがいっぱいです。YFBT/On のトグルを自分でメニューに追加してください。",
                                                                  "The menu is full. Add a toggle for YFBT/On to the menu yourself.");
            EditorUtility.SetDirty(av);
            AssetDatabase.SaveAssets();
            return warn;
        }

        // 原本是預設的 Action 層：複製一份 VRChat 預設的（表情動作照常能用）再加上我們的圖層
        static AnimatorController DefaultActionCopy(bool headFollow, bool bits)
        {
            string src = AssetDatabase.FindAssets("vrc_AvatarV3ActionLayer t:AnimatorController")
                .Select(AssetDatabase.GUIDToAssetPath).FirstOrDefault(p => p.EndsWith("/vrc_AvatarV3ActionLayer.controller"));
            if (src == null) return Build(headFollow, bits);
            EnsureFolder();
            string dst = AssetDatabase.GenerateUniqueAssetPath(Folder + "/DesktopFBT_Action.controller");
            AssetDatabase.CopyAsset(src, dst);
            var c = AssetDatabase.LoadAssetAtPath<AnimatorController>(dst);
            Populate(c, headFollow, bits);
            AssetDatabase.SaveAssets();
            return c;
        }

        static bool Ours(UnityEngine.Object c)
        {
            string p = AssetDatabase.GetAssetPath(c);
            return p.StartsWith(Folder + "/DesktopFBT_Desktop") || p.StartsWith(Folder + "/DesktopFBT_Action")
                || p.StartsWith(OldFolder + "/" + Old + "_Desktop") || p.StartsWith(OldFolder + "/" + Old + "_Action");
        }

        public static void Remove(VRCAvatarDescriptor av)
        {
            foreach (var objName in new[] { ObjectName, OldObject })
            {
                var t = av.transform.Find(objName);
                if (t != null) Undo.DestroyObjectImmediate(t.gameObject);
            }

            var layers = av.baseAnimationLayers;
            if (layers != null)
            {
                for (int i = 0; i < layers.Length; i++)
                {
                    var c = layers[i].animatorController as AnimatorController;
                    if (c == null) continue;
                    if (Ours(c))
                    {
                        Undo.RecordObject(av, "DesktopFBT Desktop");
                        layers[i].animatorController = null;
                        layers[i].isDefault = true;
                        av.baseAnimationLayers = layers;
                        string cp = AssetDatabase.GetAssetPath(c);
                        if (cp.StartsWith(Folder + "/DesktopFBT_Action") || cp.StartsWith(OldFolder + "/" + Old + "_Action")) AssetDatabase.DeleteAsset(cp);
                        continue;
                    }
                    int li = Array.FindIndex(c.layers, l => IsLayer(l.name, LayerName));
                    if (li >= 0)
                    {
                        Undo.RecordObject(c, "DesktopFBT Desktop");
                        c.RemoveLayer(li);
                        int si = Array.FindIndex(c.layers, l => IsLayer(l.name, SmoothLayer));
                        if (si >= 0) c.RemoveLayer(si);
                        int hi = Array.FindIndex(c.layers, l => IsLayer(l.name, HoldLayer));
                        if (hi >= 0) c.RemoveLayer(hi);
                        foreach (var p in c.parameters.Where(p => p.name.StartsWith(Prefix)).ToArray()) c.RemoveParameter(p);
                        foreach (var o in AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(c)))
                            if (o != null && o != c && o.name.StartsWith("YFBT ") && (o is AnimationClip || o is BlendTree))
                                UnityEngine.Object.DestroyImmediate(o, true);
                        EditorUtility.SetDirty(c);
                    }
                }
            }
            var ep = av.expressionParameters;
            if (ep != null && ep.parameters != null && ep.parameters.Any(p => p.name.StartsWith(Prefix)))
            {
                Undo.RecordObject(ep, "DesktopFBT Desktop");
                ep.parameters = ep.parameters.Where(p => !p.name.StartsWith(Prefix)).ToArray();
                EditorUtility.SetDirty(ep);
            }
            var menu = av.expressionsMenu;
            if (menu != null && menu.controls.Any(c => c.parameter != null && c.parameter.name == OnParam))
            {
                Undo.RecordObject(menu, "DesktopFBT Desktop");
                menu.controls.RemoveAll(c => c.parameter != null && c.parameter.name == OnParam);
                EditorUtility.SetDirty(menu);
            }
        }

        public static bool IsInstalled(VRCAvatarDescriptor av)
        {
            if (av.transform.Find(ObjectName) != null || av.transform.Find(OldObject) != null) return true;
            var ep = av.expressionParameters;
            return ep != null && ep.parameters != null && ep.parameters.Any(p => p.name == OnParam);
        }
    }

    public class DesktopFBTDesktopWindow : EditorWindow
    {
        VRCAvatarDescriptor avatar;
        int sync;
        int level;                  // 同期の細かさ：0 = 自動、1〜 = Profiles の番号 + 1
        bool compress;              // 他の同期パラメーターを Param Compressor に圧縮させて空きを増やす
        bool headFollow = true;
        bool useMA = true;
        string message;

        [MenuItem("Tools/デスクトップFBT/Desktop Mode (デスクトップモード)")]
        static void Open() { GetWindow<DesktopFBTDesktopWindow>("デスクトップFBT"); }

        void OnEnable()
        {
            if (avatar == null && Selection.activeGameObject != null)
                avatar = Selection.activeGameObject.GetComponentInParent<VRCAvatarDescriptor>();
        }

        void OnGUI()
        {
            int lang = EditorGUILayout.Popup("言語 / Language", Loc.En ? 1 : 0, new[] { "日本語", "English" });
            if ((lang == 1) != Loc.En) { Loc.En = lang == 1; message = null; }
            EditorGUILayout.HelpBox(Loc.T(
                "追加したあと、VRChat のメニューで「デスクトップFBT」をオンにして、デスクトップFBT アプリの「リアルタイム」で開始を押してください。VRChat のデスクトップモードでも VR でも使えます。",
                "After adding, turn on \"デスクトップFBT\" in the VRChat menu, then press Start on the \"Live sync\" page of the デスクトップFBT app. Works in VRChat desktop mode and in VR."), MessageType.Info);
            avatar = (VRCAvatarDescriptor)EditorGUILayout.ObjectField(Loc.T("アバター", "Avatar"), avatar, typeof(VRCAvatarDescriptor), true);
            sync = EditorGUILayout.Popup(Loc.T("誰に見えるか", "Visible to"), sync, Loc.En
                ? new[] { "Everyone (full body, uncompressed)", "Everyone (full body, synced in turns)", "Everyone (legs and body only)", "Only me" }
                : new[] { "全員（全身・圧縮なし）", "全員（全身・順番に同期）", "全員（脚と体のみ）", "自分のみ" });
            var mode = (DesktopRig.Sync)sync;
            if (mode == DesktopRig.Sync.EveryoneBits)
            {
                bool hasComp = DesktopRig.HasCompressor;
                using (new EditorGUI.DisabledScope(!hasComp))
                    compress = EditorGUILayout.ToggleLeft(new GUIContent(
                        Loc.T("他の同期パラメーターを圧縮して空きを増やす", "Compress other synced parameters to free space"),
                        Loc.T("アップロード時、上限を超える分だけ、他のパラメーター（衣装の切り替えなど）を順番に同期するようにします。デスクトップFBT の動きは圧縮しません",
                              "At upload, only as much as needed to fit, other parameters (outfit toggles etc.) are synced in turns. デスクトップFBT's motion is not compressed")), compress && hasComp);
                if (!hasComp)
                    EditorGUILayout.LabelField(Loc.T("　Modular Avatar（NDMF）が必要です", "  Needs Modular Avatar (NDMF)"), EditorStyles.miniLabel);
                // 同期の細かさ：自動か、最小〜最高を自分で選ぶ。細かいほど同期パラメーターを多く使う
                int count = DesktopRig.Profiles.Length;
                var names = new string[count + 1];
                names[0] = Loc.T("自動（空きに入る最高のもの）", "Auto (highest that fits)");
                for (int i = 0; i < count; i++)
                    names[i + 1] = (i + 1) + (i == 0 ? Loc.T("：最小", ": minimum") : i == count - 1 ? Loc.T("：最高", ": highest") : "") +
                                   " - " + (DesktopRig.ProfileCost(i) + 1) + " bit";
                level = Mathf.Clamp(EditorGUILayout.Popup(new GUIContent(Loc.T("同期の細かさ", "Sync detail"),
                    Loc.T("他の人から見える動きの細かさ。高いほど細かく見えますが、同期パラメーター（アバター全体で 256 bit まで）を多く使います",
                          "How fine the motion looks to other people. Higher is finer but uses more synced parameters (256 bits per avatar in total)")), level, names), 0, count);
            }
            DesktopRig.Level = mode == DesktopRig.Sync.EveryoneBits ? level - 1 : -1;
            DesktopRig.Compress = mode == DesktopRig.Sync.EveryoneBits && compress && DesktopRig.HasCompressor;
            if (mode == DesktopRig.Sync.EveryoneBits && avatar != null)
            {
                int other = DesktopRig.OtherCost(avatar);
                var alloc = DesktopRig.AllocFor(avatar);
                int limit = VRCExpressionParameters.MAX_PARAMETER_COST;
                int packed = DesktopRig.OtherPacked, steps = DesktopRig.PackSteps;
                int total = packed + DesktopRig.AllocCost(alloc), over = DesktopRig.Over();
                int best = DesktopRig.ProfileFor(DesktopRig.Budget);
                EditorGUILayout.HelpBox(
                    Loc.T("他の同期パラメーター：約 " + other + " bit", "Other synced parameters: about " + other + " bit") +
                    (avatar.customExpressions ? "" : Loc.T("（VRChat 標準の表情パラメーター）", " (VRChat's default expression parameters)")) +
                    (packed < other ? Loc.T(" → 圧縮後 約 " + packed + " bit（" + steps + " 回に分けて順番に同期）", " → about " + packed + " bit after compression (synced in " + steps + " turns)") : "") +
                    "\n" + Loc.T("全身 49 個の値：", "49 full-body values: ") + DesktopRig.Summary(alloc) +
                    "\n" + Loc.T("アバター全体：約 " + total + " / " + limit + " bit", "Whole avatar: about " + total + " / " + limit + " bit") +
                    "\n" + Loc.T("他の人には反応が速く少し粗く、自分には完全な精度で見えます。",
                                 "Other people see it responsive but slightly coarse; you see full precision."), MessageType.None);
                if (over > 0)
                    EditorGUILayout.HelpBox(
                        Loc.T("VRChat の上限（" + limit + " bit）を約 " + over + " bit 超えるため、このままではアップロードできません。他の同期パラメーターを " + over + " bit 以上減らすか、細かさを下げてください。",
                              "This exceeds VRChat's limit (" + limit + " bits) by about " + over + " bits, so the avatar cannot be uploaded like this. Reduce other synced parameters by at least " + over + " bits, or lower the detail.") +
                        (best >= 0 ? Loc.T("今の空きで選べるのは " + (best + 1) + " までです。", " With the current free space you can choose up to " + (best + 1) + ".")
                                   : Loc.T("今の空きでは最小（1）も入りません。", " With the current free space even the minimum (1) does not fit.")), MessageType.Error);
                else if (total > limit)
                    EditorGUILayout.HelpBox(
                        Loc.T("デスクトップFBT を除いても、アバターの同期パラメーターが上限（" + limit + " bit）を超えています。このままではアップロードできません。",
                              "Even without デスクトップFBT, the avatar's synced parameters exceed the limit (" + limit + " bits). The avatar cannot be uploaded like this.") +
                        (DesktopRig.Compress ? Loc.T("圧縮しても収まりません。他の同期パラメーターを減らしてください。", " Compression is not enough; reduce other synced parameters.")
                                             : Loc.T("「他の同期パラメーターを圧縮して空きを増やす」をオンにするか、他の同期パラメーターを減らしてください。",
                                                     " Turn on \"Compress other synced parameters\", or reduce other synced parameters.")), MessageType.Error);
                else if (packed < other)
                    EditorGUILayout.HelpBox(Loc.T("圧縮された他のパラメーター（衣装の切り替えなど）は、他の人には少し遅れて伝わります。回数が多いほど遅れます。デスクトップFBT の動きは圧縮されません。",
                                                  "The compressed other parameters (outfit toggles etc.) reach other people a little late; more turns means more delay. デスクトップFBT's motion is not compressed."), MessageType.Info);
                else if (level > 0 && best > level - 1)
                    EditorGUILayout.HelpBox(Loc.T("空きにはまだ余裕があります（" + (best + 1) + " まで選べます）。", "There is still free space (you can choose up to " + (best + 1) + ")."), MessageType.Info);
            }
            if (mode == DesktopRig.Sync.Everyone)
                EditorGUILayout.HelpBox(DesktopRig.HasCompressor
                    ? Loc.T("全身 49 個のパラメーターはアップロード時に Param Compressor が順番に同期します。他の人には約 1 秒遅れて、自動的に滑らかに見えます。",
                            "The 49 full-body parameters are synced in turns by the Param Compressor. Other people see the motion about 1 second late, smoothed automatically.")
                    : DesktopRig.NeedCompressor, DesktopRig.HasCompressor ? MessageType.None : MessageType.Warning);
            headFollow = EditorGUILayout.Toggle(new GUIContent(Loc.T("頭も映像に合わせる", "Head follows the video too"),
                Loc.T("オフにすると頭はマウス／HMD で動かします", "When off, the head is controlled by the mouse / headset")), headFollow);
            bool ma = DesktopRig.HasMA;
            using (new EditorGUI.DisabledScope(!ma))
                useMA = EditorGUILayout.Toggle(new GUIContent(Loc.T("Modular Avatar を使う", "Use Modular Avatar"),
                    Loc.T("アップロード時に統合します。元のアバターは変更しません", "Merged at upload time; the original avatar is left untouched")), useMA && ma);
            if (!ma)
                EditorGUILayout.HelpBox(Loc.T(
                    "Modular Avatar がありません：アバターの Action レイヤー・パラメーター・メニューを直接変更します（「削除」で元に戻せます）。",
                    "Modular Avatar is not installed: the avatar's Action layer, parameters and menu are edited directly (\"Remove\" undoes it)."), MessageType.None);

            bool blocked = mode == DesktopRig.Sync.Everyone && !DesktopRig.HasCompressor;
            using (new EditorGUI.DisabledScope(avatar == null || blocked))
            {
                bool installed = avatar != null && DesktopRig.IsInstalled(avatar);
                if (GUILayout.Button(installed ? Loc.T("追加し直す", "Add again") : Loc.T("追加", "Add"), GUILayout.Height(30)))
                    DoAdd(mode, useMA && ma);
                // ワンクリック：他のパラメーターを圧縮し、入る最高の細かさで追加する
                if (mode == DesktopRig.Sync.EveryoneBits && DesktopRig.HasCompressor &&
                    GUILayout.Button(new GUIContent(Loc.T("他のパラメーターを圧縮して、最高の細かさで追加", "Compress other parameters and add at the highest detail"),
                        Loc.T("「圧縮して空きを増やす」をオン、細かさを「自動」にして追加します", "Turns on the compression option, sets the detail to Auto, and adds")), GUILayout.Height(24)))
                {
                    compress = true;
                    level = 0;
                    DesktopRig.Compress = true;
                    DesktopRig.Level = -1;
                    DoAdd(mode, useMA && ma);
                }
            }
            using (new EditorGUI.DisabledScope(avatar == null || !DesktopRig.IsInstalled(avatar)))
                if (GUILayout.Button(Loc.T("削除", "Remove")))
                {
                    DesktopRig.Remove(avatar);
                    message = Loc.T("削除しました", "Removed");
                }
            if (!string.IsNullOrEmpty(message)) EditorGUILayout.HelpBox(message, MessageType.None);
        }

        void DoAdd(DesktopRig.Sync mode, bool withMA)
        {
            try
            {
                if (withMA)
                {
                    DesktopRig.LastWarning = null;
                    Selection.activeGameObject = DesktopRig.AddWithMA(avatar, mode, headFollow);
                    message = DesktopRig.LastWarning ?? Loc.T("追加しました（Modular Avatar）", "Added (Modular Avatar)");
                }
                else
                {
                    message = DesktopRig.ApplyDirect(avatar, mode, headFollow) ?? Loc.T("追加しました", "Added");
                }
            }
            catch (Exception e)
            {
                message = Loc.T("失敗：", "Failed: ") + e.Message;
                Debug.LogException(e);
            }
        }
    }
}
#endif
