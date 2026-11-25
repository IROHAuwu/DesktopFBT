"""給別人看的同步（第 2 版）：分配表和 Unity 工具一致、量化有 0 這一格、沿著骨頭補誤差、實際送出的 Bool 還原得回來。
執行：python tests/test_sync.py"""
import json
import math
import os
import re
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import desktop as DK  # noqa: E402
import motion  # noqa: E402
import sync as SY  # noqa: E402
import sync_profiles as SP  # noqa: E402

CS = open(os.path.join(ROOT, 'unity', 'Desktop', 'DesktopFBTDesktop.cs'), encoding='utf-8').read()


def frames(n=60):
    """模擬動作的參數（走路＋手的動作），和即時模式送出的一樣。"""
    dp = DK.DesktopPose(grounded=True)
    ok = np.ones(len(motion.person(0.0)), bool)
    return [dp(motion.person(i / 10.0), ok) for i in range(n)]


def test_tables_match_unity():
    block = CS[CS.index('int[][] Profiles = {'):CS.index('static readonly int[] FollowFrom')]
    rows = [[int(x) for x in r.split(',')] for r in re.findall(r'new\[\] \{([^}]*)\}', block)]
    assert rows == [[p.get(n, 0) for n in SY.NAMES] for p in SY.PROFILES]
    nums = lambda name: [int(x) for x in re.search(name + r' = \{([^}]*)\}', CS).group(1).split(',')]      # noqa: E731
    assert dict(zip(nums('FollowFrom'), nums('FollowTo'))) == SY.FOLLOW
    assert list(SY.NAMES) == [f'm{i}' for i in DK.MUSCLES] + list(DK.BODY_PARAMS)      # Unity 的 FloatParams() 就是這個順序
    assert 'lo = -1f; hi = 1f; zero = true;' in CS and SY.RANGE.get('m0', (-1.0, 1.0, True)) == (-1.0, 1.0, True)
    assert '"TiltX" || p == Prefix + "TiltZ") { lo = -0.5f; hi = 0.5f; }' in CS and SY.RANGE['TiltX'] == SY.RANGE['TiltZ'] == (-0.5, 0.5, True)
    assert '"Y") { lo = -0.9f; hi = 0.5f; zero = false; }' in CS and SY.RANGE['Y'] == (-0.9, 0.5, False)
    assert SY.RANGE['YawS'] == SY.RANGE['YawC'] == (-1.0, 1.0, True) and set(SY.RANGE) == set(DK.BODY_PARAMS)
    assert '(vmax - vmin) / ((1 << n) - (zero ? 2 : 1))' in CS and 'step * (1 << (n - 1 - i))' in CS
    assert 'Prefix + "P" + Profile + "/" + p.Substring(Prefix.Length) + "_" + i' in CS and SY.bit_name(2, 'm0', 1) == 'P2/m0_1'
    assert 'Profile = Level < 0 ? ProfileFor(Budget) : Mathf.Min(Level, Profiles.Length - 1);' in CS      # 細かさは自動か手動（上限を超える選択は警告）
    assert 'ProfileCost(Profile) - Budget' in CS and 'MessageType.Error' in CS
    # 其他參數交給 Param Compressor（同梱的或 lilMCR 的）壓縮：我們的 Bool 要排除；估算用的常數和 lilMCR 的 MakePlan 一樣（每次最多 8 個數值、64 個 Bool）
    assert 'Mathf.Clamp(Mathf.CeilToInt((float)numbers / steps), 1, 8)' in CS and 'Mathf.Clamp(Mathf.CeilToInt((float)bools / steps), 1, 64)' in CS
    assert 'if (on) list.Add(Prefix);' in CS and 'SetOurExclusion(have, false);' in CS and '"liltoon_MCR.MacaronParamCompressorPass"' in CS
    assert SY.COST == sorted(SY.COST) and SY.COST[0] >= 48 and SY.COST[-1] <= 255 - 24      # VRChat 預設的參數佔 24 bit 時最大的那種還放得下
    for p in SY.PROFILES:
        assert all(2 <= n <= DK.MAX_BITS for n in p.values()) and set(p) <= set(SY.NAMES)
        assert all(b in p for b in SY.BODY)                                             # 朝向、傾斜、高度每一種都要同步（沒同步的話別人那邊是 0：身體矮一截）
        assert not any(k[0] == 'm' and int(k[1:]) in SY.FOLLOW for k in p)              # 跟著別人的肌肉沒有自己的 bit
        assert all(p.get(f'm{a}', 0) == p.get(f'm{b}', 0) for a, b in SY._MIRROR.items())   # 左右一樣
    print(f'  分配表 {SY.COST} bit，和 Unity 工具一致')


def test_levels():
    for name in ('m24', 'TiltX', 'YawS', 'Y'):
        lo, hi, zero = SY.RANGE.get(name, (-1.0, 1.0, True))
        for n in range(2, 7):
            _, step, count = SY.levels(name, n)
            assert count == (1 << n) - (1 if zero else 0) and abs(lo + (count - 1) * step - hi) < 1e-12
            for v in np.linspace(lo - 0.2, hi + 0.2, 301):
                q = SY.quantize(name, n, float(v))
                b = SY.bits(q, n)
                assert 0 <= q < count and len(b) == n and SY.unbits(b) == q
                unity = lo + sum(step * (1 << (n - 1 - k)) for k in range(n) if b[k])      # Unity 的還原方式
                assert abs(unity - SY.value(name, n, q)) < 1e-9
                assert abs(unity - min(max(v, lo), hi)) <= step / 2 + 1e-9, (name, n, v)
            if zero:
                assert SY.value(name, n, SY.quantize(name, n, 0.0)) == 0.0                  # 0 剛好有一格
            assert abs(SY.value(name, n, SY.quantize(name, n, hi)) - hi) < 1e-12 and SY.value(name, n, SY.quantize(name, n, lo)) == lo
    print('  量化：0、最大、最小都剛好有一格；和 Unity 的還原公式一致')


def test_upright_is_upright():
    """站直不動：別人看到的身體不歪（第 1 版在 4 bit 時側傾 11°）。"""
    still = {n: 0.0 for n in SY.NAMES}
    still.update(YawC=1.0, Y=0.27)
    for i, p in enumerate(SY.PROFILES):
        codes, vals = SY.remote(still, p)
        assert set(vals) == set(SY.NAMES) and set(codes) == set(p)
        assert vals['TiltX'] == vals['TiltZ'] == vals['YawS'] == 0.0 and vals['YawC'] == 1.0
        assert abs(vals['Y'] - 0.27) * 75 < 3.5 and all(vals[f'm{k}'] == 0.0 for k in DK.MUSCLES), i
    old = DK.decode(DK.encode(0.0)[:4])
    assert abs(old * 180 - 11.25) < 1e-9
    print('  站直：新的同步傾斜 0°（舊的 4 bit 時 11.25°）')


def test_chain_compensation():
    """別人看到的姿勢：bit 越多越準；同樣的 bit 數比第 1 版準很多；跟著的骨頭和帶頭的一樣。"""
    fr = frames()
    refs = [SP.pose(f) for f in fr]

    def err(vals):
        e = []
        for (P0, _), v in zip(refs, vals):
            P1, _ = SP.pose(v)
            e.append(np.mean([100 * math.dist(P0[n], P1[n]) for n in ('Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')]))
        return float(np.mean(e))
    errs = []
    for p in SY.PROFILES:
        vals = [SY.remote(f, p)[1] for f in fr]
        for v in vals:
            assert all(v[f'm{a}'] == v[f'm{b}'] for a, b in SY.FOLLOW.items())
            assert all(-1.0 <= v[n] <= 1.0 for n in SY.NAMES)
        errs.append(err(vals))
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    assert errs[0] < 16 and errs[2] < 8 and errs[-1] < 3, errs
    old4 = err([{k: DK.decode(DK.encode(v)[:4]) for k, v in f.items()} for f in fr])      # 第 1 版，49 個值各 4 bit = 196 bit
    assert errs[2] < 0.8 * old4 and errs[3] < 0.6 * old4, (errs, old4)
    print('  頭手腳的位置誤差（公分）：' + '、'.join(f'{c} bit {e:.1f}' for c, e in zip(SY.COST, errs)) + f'；舊的 196 bit {old4:.1f}')


def test_sent_bits():
    """實際送出去的 OSC：Float、第 1 版的 Bool、每一種分配表的 Bool；用送出的 Bool 還原 = remote() 算的。"""
    class Client:
        def __init__(self):
            self.msgs = {}

        def send_message(self, addr, v):
            self.msgs[addr] = v
    osc = DK.AvatarOSC.__new__(DK.AvatarOSC)
    osc.client, osc.sync, osc.sent, osc.last, osc.n = Client(), SY, {}, {}, 0
    f = frames(8)[5]
    pre = '/avatar/parameters/' + DK.PREFIX
    slots = len(SY.PROFILES) + 1
    per_call = []
    for _ in range(slots):                                         # 輪流：一次一種，一輪下來全部送過
        osc.client.msgs = {}
        osc.send(f)
        per_call.append(len(osc.client.msgs))
        assert all(abs(osc.client.msgs[pre + n] - f[n]) < 1e-12 for n in SY.NAMES)      # Float 每次都送
        osc.client.all = dict(getattr(osc.client, 'all', {}), **osc.client.msgs)
    m = osc.client.all
    assert per_call == [49 + 49 * DK.MAX_BITS] + [49 + c for c in SY.COST]
    assert all([m[pre + f'{n}_{k}'] for k in range(DK.MAX_BITS)] == DK.encode(f[n]) for n in SY.NAMES)      # 舊同步的內容沒變
    for i, p in enumerate(SY.PROFILES):
        codes, _ = SY.remote(f, p)
        for name, n in p.items():
            got = SY.unbits([m[pre + SY.bit_name(i, name, k)] for k in range(n)])
            assert got == codes[name][0], (i, name)
    assert len(m) == 49 + 49 * DK.MAX_BITS + sum(SY.COST)
    osc.client.msgs = {}
    for _ in range(slots):
        osc.send(f)                                                # 再一輪：同步的 Bool 沒變 → 只有 Float
    assert set(osc.client.msgs) == {pre + n for n in SY.NAMES}
    for _ in range(int(osc.REFRESH_S * osc.RATE)):                 # 每隔幾秒全部重送一次
        osc.send(f)
    assert len(osc.client.msgs) == len(m)
    print(f'  送出 {len(m)} 個位址（Float 49、舊同步 {49 * DK.MAX_BITS}、新同步 {sum(SY.COST)}），一次最多 {max(per_call)} 個')


def test_bad_numbers():
    """追蹤偶爾算出 NaN／無限大時：不送出去（VRChat 那邊的姿勢會壞掉）、也不能丟例外（送出的執行緒停掉 Avatar 就不動了）。"""
    class Client:
        def __init__(self):
            self.msgs = []

        def send_message(self, addr, v):
            self.msgs.append((addr, v))
    osc = DK.AvatarOSC.__new__(DK.AvatarOSC)
    osc.client, osc.sync, osc.sent, osc.last, osc.n = Client(), SY, {}, {}, 0
    good = frames(4)[2]
    pre = '/avatar/parameters/' + DK.PREFIX
    for _ in range(7):
        osc.send(good)
    for bad in (float('nan'), float('inf'), float('-inf')):
        f = dict(good, m24=bad, TiltX=bad, YawS=bad)
        osc.client.msgs = []
        for _ in range(14):                                         # 兩輪：每一種同步都經過
            osc.send(f)
        floats = [v for _, v in osc.client.msgs if isinstance(v, float)]
        assert floats and all(math.isfinite(v) for v in floats)
        assert all(v == good['m24'] for a, v in osc.client.msgs if a == pre + 'm24')      # 沿用上一次好的值
    short = {k: v for k, v in good.items() if k != 'm24'}          # 少一個值：同步的部分這次不送，其他照常
    osc.send(short)
    print('  NaN／無限大／缺值：不送壞數字、不丟例外')


def test_old_sync_notice():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, 'usr_x', 'Avatars'))
    f = os.path.join(d, 'usr_x', 'Avatars', 'avtr_1.json')

    def write(names):
        with open(f, 'w', encoding='utf-8') as fp:
            json.dump({'name': 'Mine', 'parameters': [{'name': n} for n in names]}, fp)
    write(['YFBT/TiltX', 'YFBT/TiltX_0', 'YFBT/m0_0'])
    assert DK.old_sync_avatar(d) == 'Mine' and DK.stale_avatar(d) is None
    write(['YFBT/TiltX', 'YFBT/P2/TiltX_0', 'YFBT/P2/m0_0'])
    assert DK.old_sync_avatar(d) is None
    write(['YFBT/TiltX', 'YFBT/m0'])                               # 只有自己看得到的設定：沒有同步的 Bool，不用提醒
    assert DK.old_sync_avatar(d) is None
    write(['Other'])
    assert DK.old_sync_avatar(d) is None
    print('  舊同步方式的 Avatar 會提醒')


def main():
    for f in (test_tables_match_unity, test_levels, test_upright_is_upright, test_chain_compensation, test_sent_bits, test_bad_numbers, test_old_sync_notice):
        print(f.__name__)
        f()
    print('OK')


if __name__ == '__main__':
    main()
