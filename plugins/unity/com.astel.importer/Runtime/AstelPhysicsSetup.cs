// Reads Astel's engine.json sidecar and auto-configures Unity physics on a GameObject.
//
// Usage:
//   var setup = new AstelPhysicsSetup(manifest, gameObject);
//   setup.Apply();
//
// What it does:
//   • Adds a Rigidbody with mass and centre-of-mass from L5 mass_props.
//   • Creates PhysicMaterial from L6 region[0] friction + restitution.
//   • Logs articulation hints from L6 (auto-configuring ArticulationBody is a
//     Unity version-specific follow-on; hints are available for manual setup).
//
// engine.json is a sibling DELIVERY artifact (download it next to the splat
// file it names in `splat_file`, e.g. l3.spz) — it is NOT a member inside the
// package.astel zip. The producer assembles it from the real L5/L6 layers.
//
// Coordinate convention: data is in the Astel 3DGS frame (right-handed, +Y up,
// metres). Unity is left-handed, +Y up, +Z forward. The X-axis flip
// (pos_unity.x = −pos.x) is applied to centre_of_mass.
// See docs/architecture/coordinate-conventions.md.

using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Astel
{
    public class AstelPhysicsSetup
    {
        readonly AstelManifest _manifest;
        readonly GameObject _root;

        public AstelPhysicsSetup(AstelManifest manifest, GameObject root)
        {
            _manifest = manifest;
            _root = root;
        }

        public void Apply()
        {
            ApplyRigidbody();
            ApplyPhysicMaterial();
            LogArticulationHints();
        }

        // ---------------------------------------------------------------
        // Rigidbody: mass + COM from L5
        // ---------------------------------------------------------------
        void ApplyRigidbody()
        {
            var l5 = _manifest.l5;
            if (l5 == null || l5.mass_props == null) return;

            var rb = _root.GetComponent<Rigidbody>() ?? _root.AddComponent<Rigidbody>();
            var mp = l5.mass_props;

            rb.mass = mp.mass_kg > 0f ? mp.mass_kg : 1f;

            if (mp.center_of_mass != null && mp.center_of_mass.Length == 3)
            {
                // Apply X-flip for right-hand→left-hand and scale by mpu.
                float mpu = _manifest.meters_per_unit;
                rb.centerOfMass = new Vector3(
                    -mp.center_of_mass[0] * mpu,
                     mp.center_of_mass[1] * mpu,
                     mp.center_of_mass[2] * mpu
                );
            }
        }

        // ---------------------------------------------------------------
        // PhysicMaterial: friction + restitution from L6 region 0
        // ---------------------------------------------------------------
        void ApplyPhysicMaterial()
        {
            var l6 = _manifest.l6;
            if (l6 == null || l6.regions == null || l6.regions.Count == 0) return;

            var region = l6.regions[0];
            var mat = new PhysicMaterial($"Astel_{region.name}")
            {
                dynamicFriction = region.friction,
                staticFriction  = region.friction,
                bounciness      = region.restitution,
                frictionCombine = PhysicMaterialCombine.Average,
                bounceCombine   = PhysicMaterialCombine.Average,
            };

            // Apply to all colliders on the root and children
            foreach (var col in _root.GetComponentsInChildren<Collider>())
                col.material = mat;
        }

        // ---------------------------------------------------------------
        // Articulation hints — logged for manual ArticulationBody setup
        // ---------------------------------------------------------------
        void LogArticulationHints()
        {
            var l6 = _manifest.l6;
            if (l6 == null || l6.articulation == null) return;

            foreach (var hint in l6.articulation)
            {
                Debug.Log(
                    $"[Astel] Articulation hint: regions {hint.region_a}↔{hint.region_b} " +
                    $"joint={hint.joint_type}. " +
                    $"Add ArticulationBody manually or use AstelArticulationSetup."
                );
            }
        }

        // ---------------------------------------------------------------
        // Static helpers for reading the engine.json sidecar
        // ---------------------------------------------------------------

        //: The flat physics-setup sidecar name the producer emits.
        public const string EngineSetupFile = "engine.json";

        /// Load engine.json from a directory of downloaded artifacts (the folder
        /// also holds the splat named by AstelManifest.splat_file).
        public static AstelManifest LoadFromDirectory(string dir)
        {
            string path = Path.Combine(dir, EngineSetupFile);
            if (!File.Exists(path))
                throw new FileNotFoundException(EngineSetupFile + " not found in " + dir, path);
            return JsonUtility.FromJson<AstelManifest>(File.ReadAllText(path));
        }

        /// Load engine.json directly from its file path.
        public static AstelManifest LoadEngineSetup(string engineJsonPath)
        {
            return JsonUtility.FromJson<AstelManifest>(File.ReadAllText(engineJsonPath));
        }
    }
}
