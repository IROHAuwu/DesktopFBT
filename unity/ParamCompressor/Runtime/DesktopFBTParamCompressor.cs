// デスクトップFBT Param Compressor — settings component.
// Taken from lilMCR Param Compressor (IROHAuwu/lilMCR), which is based on ND Parameter Compressor (sakuraimikoto33, MIT)
// and Laura's Param Compressor (MIT). See THIRD_PARTY_NOTICES.
using UnityEngine;
using VRC.SDKBase;

namespace DesktopFBT
{
    /// <summary>
    /// デスクトップFBT Param Compressor (non-destructive). The same component as lilMCR Param Compressor, bundled so that
    /// デスクトップFBT works without lilMCR. The デスクトップFBT tool adds it; nothing happens in the
    /// editor: while uploading, after Modular Avatar has merged everything, the NDMF pass checks the synced parameter
    /// cost and, only when it is over VRChat's limit, makes part of the synced parameters take turns on a few shared
    /// sync parameters. Avatars that are already within the limit are not changed. Delete the object to undo.
    ///
    /// Based on ND Parameter Compressor (sakuraimikoto33, MIT), itself based on Laura's Param Compressor (MIT).
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("デスクトップFBT/Param Compressor")]
    public class DesktopFBTParamCompressor : MonoBehaviour, IEditorOnly
    {
        [Tooltip("Upper limit of sync steps (turns). The fewest steps that fit within the limit are used; more steps = more delay for other players.")]
        [Range(2, 32)]
        public int maxSyncSteps = 8;

        [Tooltip("Never compress Bool parameters.")]
        public bool excludeBools;
        [Tooltip("Never compress Int parameters.")]
        public bool excludeInts;
        [Tooltip("Never compress Float parameters.")]
        public bool excludeFloats;

        [Tooltip("Parameters whose name starts with one of these are never compressed.")]
        public string[] excludedPrefixes = new string[0];
        [Tooltip("Parameters whose name ends with one of these are never compressed.")]
        public string[] excludedSuffixes = new string[0];
    }
}
