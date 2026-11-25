// DesktopFBT — BVH → Unity Humanoid 動畫
// 把 .bvh 放進 Assets（或匯入 DesktopFBT 產生的 .unitypackage），Unity 會自動轉成 Humanoid 的 AnimationClip，
// 任何 Humanoid Avatar（包含 VRChat 的 Avatar）都能直接使用。
// 需要的動畫在 .bvh 底下：選取後按 Ctrl+D 可以複製成獨立的 .anim 檔。
// 轉換用的是 Unity 自己的 HumanPoseHandler，所以肌肉值和 Unity 的計算方式一致。
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEditor;
#if UNITY_2020_2_OR_NEWER
using UnityEditor.AssetImporters;
#else
using UnityEditor.Experimental.AssetImporters;
#endif
using UnityEngine;

namespace DesktopFBT
{
    [ScriptedImporter(3, "bvh")]
    public class DesktopFBTBvhImporter : ScriptedImporter
    {
        [Tooltip("ループ再生（ダンスや待機モーション向け） / Loop (recommended for dances and idle motions)")]
        public bool loop = true;

        [Tooltip("原点から開始する（動画での最初の位置を取り除く） / Start at the origin (removes the starting position in the video)")]
        public bool startAtOrigin = true;

        [Tooltip("その場で動く：水平方向の移動を取り除き、体の動きだけ残す / In place: removes horizontal travel, keeps only the body motion")]
        public bool inPlace = false;

        [Tooltip("最初の何秒だけ使うか（0 = 全部） / Keep only this many seconds from the start (0 = all)")]
        public float maxSeconds = 0f;

        // BVH 骨頭名稱 → Unity Humanoid 骨頭
        static readonly Dictionary<string, string> HumanMap = new Dictionary<string, string>
        {
            {"Hips", "Hips"}, {"Spine", "Spine"}, {"Chest", "Chest"}, {"Neck", "Neck"}, {"Head", "Head"},
            {"LeftShoulder", "LeftShoulder"}, {"LeftArm", "LeftUpperArm"}, {"LeftForeArm", "LeftLowerArm"}, {"LeftHand", "LeftHand"},
            {"RightShoulder", "RightShoulder"}, {"RightArm", "RightUpperArm"}, {"RightForeArm", "RightLowerArm"}, {"RightHand", "RightHand"},
            {"LeftUpLeg", "LeftUpperLeg"}, {"LeftLeg", "LeftLowerLeg"}, {"LeftFoot", "LeftFoot"}, {"LeftToeBase", "LeftToes"},
            {"RightUpLeg", "RightUpperLeg"}, {"RightLeg", "RightLowerLeg"}, {"RightFoot", "RightFoot"}, {"RightToeBase", "RightToes"},
        };

        class Joint
        {
            public string name;
            public Joint parent;
            public Vector3 offset;          // BVH 座標（公分）
            public string[] channels;
            public int channelStart;
            public Transform tf;
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            string text = File.ReadAllText(ctx.assetPath);
            List<Joint> joints;
            float[][] frames;
            float frameTime;
            try
            {
                Parse(text, out joints, out frames, out frameTime);
            }
            catch (Exception e)
            {
                ctx.LogImportError("DesktopFBT: BVH の形式エラー / BVH format error: " + e.Message);
                return;
            }
            if (frames.Length == 0 || joints.Count == 0)
            {
                ctx.LogImportError("DesktopFBT: BVH にフレームがありません / the BVH has no frames");
                return;
            }
            int count = frames.Length;
            if (maxSeconds > 0f)
                count = Mathf.Clamp(Mathf.CeilToInt(maxSeconds / frameTime) + 1, 1, frames.Length);

            // 建立 T-pose 的暫時骨架（BVH 是右手座標：+X = 角色左邊；Unity 是左手座標：+X = 右邊 → X 鏡像）
            var root = new GameObject("DesktopFBT_Rig") { hideFlags = HideFlags.HideAndDontSave };
            try
            {
                float minY = 0f;
                var restWorld = new Dictionary<Joint, Vector3>();
                foreach (var j in joints)
                {
                    var go = new GameObject(j.name) { hideFlags = HideFlags.HideAndDontSave };
                    j.tf = go.transform;
                    j.tf.SetParent(j.parent == null ? root.transform : j.parent.tf, false);
                    j.tf.localPosition = ToUnity(j.offset);
                    j.tf.localRotation = Quaternion.identity;
                    Vector3 w = (j.parent == null ? Vector3.zero : restWorld[j.parent]) + ToUnity(j.offset);
                    restWorld[j] = w;
                    minY = Mathf.Min(minY, w.y);
                }
                Joint hips = joints[0];
                // 靜止姿勢：腳底在地面、骨盆在腿長的高度（Humanoid 需要這樣才能算出正確的比例）
                hips.tf.localPosition = new Vector3(0f, -minY + 0.02f, 0f);

                var desc = new HumanDescription();
                var human = new List<HumanBone>();
                foreach (var j in joints)
                {
                    string hn;
                    if (!HumanMap.TryGetValue(j.name, out hn)) continue;
                    var lim = new HumanLimit();
                    lim.useDefaultValues = true;
                    var hb = new HumanBone();
                    hb.boneName = j.name;
                    hb.humanName = hn;
                    hb.limit = lim;
                    human.Add(hb);
                }
                desc.human = human.ToArray();
                var skel = new List<SkeletonBone>();
                foreach (var t in root.GetComponentsInChildren<Transform>(true))
                    skel.Add(new SkeletonBone { name = t.name, position = t.localPosition, rotation = t.localRotation, scale = t.localScale });
                desc.skeleton = skel.ToArray();
                desc.upperArmTwist = 0.5f;
                desc.lowerArmTwist = 0.5f;
                desc.upperLegTwist = 0.5f;
                desc.lowerLegTwist = 0.5f;
                desc.armStretch = 0.05f;
                desc.legStretch = 0.05f;
                desc.feetSpacing = 0f;
                desc.hasTranslationDoF = false;

                Avatar avatar = AvatarBuilder.BuildHumanAvatar(root, desc);
                if (avatar == null || !avatar.isValid || !avatar.isHuman)
                {
                    ctx.LogImportError("DesktopFBT: Humanoid の骨格を作れません（BVH のボーン名が合いません） / could not build a Humanoid rig (BVH bone names do not match)");
                    return;
                }
                avatar.name = Path.GetFileNameWithoutExtension(ctx.assetPath) + "Avatar";

                // 只寫身體的肌肉（手指、眼睛、下巴沒有資料，維持 Avatar 原本的樣子）
                var muscles = new List<int>();
                for (int m = 0; m < HumanTrait.MuscleCount; m++)
                {
                    int bone = HumanTrait.BoneFromMuscle(m);
                    if (bone >= 0 && bone <= (int)HumanBodyBones.RightToes) muscles.Add(m);
                }
                var muscleCurves = new AnimationCurve[HumanTrait.MuscleCount];
                foreach (int m in muscles) muscleCurves[m] = new AnimationCurve();
                var rootT = new[] { new AnimationCurve(), new AnimationCurve(), new AnimationCurve() };
                var rootQ = new[] { new AnimationCurve(), new AnimationCurve(), new AnimationCurve(), new AnimationCurve() };

                var handler = new HumanPoseHandler(avatar, root.transform);
                var pose = new HumanPose();
                Vector3 start = Vector3.zero;
                Quaternion prevQ = Quaternion.identity;
                try
                {
                    for (int f = 0; f < count; f++)
                    {
                        float[] row = frames[f];
                        foreach (var j in joints)
                        {
                            float rz = 0f, rx = 0f, ry = 0f;
                            string order = "";
                            bool hasPos = false;
                            Vector3 p = Vector3.zero;
                            for (int c = 0; c < j.channels.Length; c++)
                            {
                                float v = row[j.channelStart + c];
                                switch (j.channels[c])
                                {
                                    case "Xposition": p.x = v; hasPos = true; break;
                                    case "Yposition": p.y = v; hasPos = true; break;
                                    case "Zposition": p.z = v; hasPos = true; break;
                                    case "Xrotation": rx = v; order += "X"; break;
                                    case "Yrotation": ry = v; order += "Y"; break;
                                    case "Zrotation": rz = v; order += "Z"; break;
                                }
                            }
                            if (hasPos && j.parent == null) j.tf.localPosition = ToUnity(p);
                            j.tf.localRotation = ToUnity(EulerRH(order, rx, ry, rz));
                        }
                        handler.GetHumanPose(ref pose);
                        float t = f * frameTime;
                        if (f == 0) start = pose.bodyPosition;
                        Vector3 bp = pose.bodyPosition;
                        if (startAtOrigin) { bp.x -= start.x; bp.z -= start.z; }
                        if (inPlace) { bp.x = startAtOrigin ? 0f : start.x; bp.z = startAtOrigin ? 0f : start.z; }
                        Quaternion q = pose.bodyRotation;
                        if (f > 0 && Quaternion.Dot(prevQ, q) < 0f) q = new Quaternion(-q.x, -q.y, -q.z, -q.w);   // 四元數連續
                        prevQ = q;
                        rootT[0].AddKey(t, bp.x); rootT[1].AddKey(t, bp.y); rootT[2].AddKey(t, bp.z);
                        rootQ[0].AddKey(t, q.x); rootQ[1].AddKey(t, q.y); rootQ[2].AddKey(t, q.z); rootQ[3].AddKey(t, q.w);
                        foreach (int m in muscles) muscleCurves[m].AddKey(t, pose.muscles[m]);
                    }
                }
                finally
                {
                    handler.Dispose();
                }

                var clip = new AnimationClip();
                clip.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
                clip.frameRate = Mathf.Max(1f, Mathf.Round(1f / frameTime));
                string[] tn = { "RootT.x", "RootT.y", "RootT.z" };
                string[] qn = { "RootQ.x", "RootQ.y", "RootQ.z", "RootQ.w" };
                for (int i = 0; i < 3; i++) clip.SetCurve("", typeof(Animator), tn[i], rootT[i]);
                for (int i = 0; i < 4; i++) clip.SetCurve("", typeof(Animator), qn[i], rootQ[i]);
                foreach (int m in muscles)
                    clip.SetCurve("", typeof(Animator), HumanTrait.MuscleName[m], muscleCurves[m]);

                var s = AnimationUtility.GetAnimationClipSettings(clip);
                s.loopTime = loop;
                s.loopBlend = false;
                s.keepOriginalOrientation = true;
                s.keepOriginalPositionY = true;
                s.keepOriginalPositionXZ = true;
                s.loopBlendOrientation = true;      // 「Bake Into Pose」：不會讓 Avatar 本身旋轉或往下沉
                s.loopBlendPositionY = true;
                s.loopBlendPositionXZ = inPlace;
                AnimationUtility.SetAnimationClipSettings(clip, s);

                ctx.AddObjectToAsset("clip", clip);
                ctx.AddObjectToAsset("avatar", avatar);
                ctx.SetMainObject(clip);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // ---------------------------------------------------------------- 座標轉換
        static Vector3 ToUnity(Vector3 p)
        {
            return new Vector3(-p.x, p.y, p.z) * 0.01f;
        }

        static Quaternion ToUnity(Quaternion q)
        {
            return new Quaternion(q.x, -q.y, -q.z, q.w);
        }

        static Quaternion AxisAngleRH(int axis, float deg)
        {
            float h = deg * Mathf.Deg2Rad * 0.5f, s = Mathf.Sin(h), c = Mathf.Cos(h);
            if (axis == 0) return new Quaternion(s, 0f, 0f, c);
            if (axis == 1) return new Quaternion(0f, s, 0f, c);
            return new Quaternion(0f, 0f, s, c);
        }

        // BVH：通道順序由外到內，例如 "ZXY" → R = Rz · Rx · Ry
        static Quaternion EulerRH(string order, float x, float y, float z)
        {
            Quaternion q = Quaternion.identity;
            foreach (char ch in order)
            {
                if (ch == 'X') q = q * AxisAngleRH(0, x);
                else if (ch == 'Y') q = q * AxisAngleRH(1, y);
                else q = q * AxisAngleRH(2, z);
            }
            return q;
        }

        // ---------------------------------------------------------------- BVH 讀取
        static void Parse(string text, out List<Joint> joints, out float[][] frames, out float frameTime)
        {
            var tok = text.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var inv = CultureInfo.InvariantCulture;
            joints = new List<Joint>();
            var stack = new Stack<Joint>();
            int i = 0, channels = 0;
            bool endSite = false;
            int endDepth = 0;
            while (i < tok.Length && tok[i] != "MOTION")
            {
                string t = tok[i];
                if (endSite)
                {
                    if (t == "{") endDepth++;
                    else if (t == "}") { endDepth--; if (endDepth == 0) endSite = false; }
                    i++;
                    continue;
                }
                if (t == "ROOT" || t == "JOINT")
                {
                    var j = new Joint { name = tok[i + 1], parent = stack.Count > 0 ? stack.Peek() : null, channels = new string[0] };
                    joints.Add(j);
                    stack.Push(j);
                    i += 2;
                }
                else if (t == "End")
                {
                    endSite = true;
                    endDepth = 0;
                    i += 2;
                }
                else if (t == "OFFSET")
                {
                    stack.Peek().offset = new Vector3(float.Parse(tok[i + 1], inv), float.Parse(tok[i + 2], inv), float.Parse(tok[i + 3], inv));
                    i += 4;
                }
                else if (t == "CHANNELS")
                {
                    int n = int.Parse(tok[i + 1], inv);
                    var j = stack.Peek();
                    j.channels = new string[n];
                    for (int c = 0; c < n; c++) j.channels[c] = tok[i + 2 + c];
                    j.channelStart = channels;
                    channels += n;
                    i += 2 + n;
                }
                else if (t == "}")
                {
                    stack.Pop();
                    i++;
                }
                else i++;
            }
            // MOTION / Frames: N / Frame Time: x
            int frameCount = int.Parse(tok[i + 2], inv);
            frameTime = float.Parse(tok[i + 5], inv);
            if (frameTime <= 0f) frameTime = 1f / 30f;
            i += 6;
            frames = new float[frameCount][];
            for (int f = 0; f < frameCount; f++)
            {
                var row = new float[channels];
                for (int c = 0; c < channels; c++) row[c] = float.Parse(tok[i++], inv);
                frames[f] = row;
            }
        }
    }
}
#endif
