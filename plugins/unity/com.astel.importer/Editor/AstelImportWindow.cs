// Editor window: pick a folder of downloaded Astel artifacts → auto-configure physics.
// Window path: Astel → Import Asset
//
// Flow:
//   1. User picks the folder holding a generation's downloaded artifacts (it
//      contains engine.json + the splat it names, e.g. l3.spz).
//   2. engine.json is read (AstelPhysicsSetup.LoadFromDirectory).
//   3. AstelPhysicsSetup.Apply() configures Rigidbody + PhysicMaterial on a new GO.
//   4. The splat named in engine.json is logged for import via UnityGaussianSplatting.
//
// engine.json + the splat are SIBLING delivery artifacts (served by the API per
// name); they are not members inside the package.astel zip, so this window does
// not extract a package — it reads the downloaded folder directly.

#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Astel.Editor
{
    public class AstelImportWindow : EditorWindow
    {
        string _artifactDir = "";

        [MenuItem("Astel/Import Asset")]
        static void Open() => GetWindow<AstelImportWindow>("Astel Import").Show();

        void OnGUI()
        {
            GUILayout.Label("Import Astel Artifacts", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            _artifactDir = EditorGUILayout.TextField("Artifact folder", _artifactDir);
            if (GUILayout.Button("Browse…"))
                _artifactDir = EditorUtility.OpenFolderPanel("Select artifact folder", "", "");

            EditorGUILayout.Space();

            bool hasEngine = !string.IsNullOrEmpty(_artifactDir) &&
                File.Exists(Path.Combine(_artifactDir, AstelPhysicsSetup.EngineSetupFile));
            GUI.enabled = hasEngine;
            if (GUILayout.Button("Import"))
                DoImport(_artifactDir);
            GUI.enabled = true;

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Pick the folder with a generation's downloaded artifacts " +
                "(engine.json + the splat it names).\n" +
                "Requires UnityGaussianSplatting (aras-p) for splat rendering.\n" +
                "Collision, mass, and material are auto-configured from the L5/L6 layers.",
                MessageType.Info
            );
        }

        static void DoImport(string artifactDir)
        {
            // 1. Load engine.json from the downloaded-artifact folder.
            AstelManifest manifest;
            try { manifest = AstelPhysicsSetup.LoadFromDirectory(artifactDir); }
            catch (FileNotFoundException ex)
            {
                EditorUtility.DisplayDialog("Astel", ex.Message, "OK");
                return;
            }

            // 2. Create a root GameObject named after the folder.
            string assetId = new DirectoryInfo(artifactDir).Name;
            var go = new GameObject(assetId);
            Undo.RegisterCreatedObjectUndo(go, "Import Astel Asset");

            // 3. Auto-configure physics from L5/L6.
            new AstelPhysicsSetup(manifest, go).Apply();

            // 4. Log the splat file (named in engine.json) for manual GS import.
            string splatName = string.IsNullOrEmpty(manifest.splat_file)
                ? "l3.spz" : manifest.splat_file;
            string splatPath = Path.Combine(artifactDir, splatName);
            if (!File.Exists(splatPath))
            {
                string ply = Path.Combine(artifactDir, "l3.ply");
                if (File.Exists(ply)) splatPath = ply;
            }
            Debug.Log($"[Astel] Splat file: {splatPath}\n" +
                      "Drag it into a GaussianSplatAsset via UnityGaussianSplatting, " +
                      "then attach GaussianSplatRenderer to this GameObject.");

            Selection.activeGameObject = go;
            EditorGUIUtility.PingObject(go);
        }
    }
}
#endif
