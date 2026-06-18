// NUnit tests for AstelPhysicsSetup — pure logic, no engine rendering needed.
// Run in Unity Test Runner (EditMode).

using NUnit.Framework;
using UnityEngine;
using Astel;

namespace Astel.Tests
{
    [TestFixture]
    public class AstelPhysicsSetupTests
    {
        GameObject _go;

        [SetUp]
        public void SetUp() => _go = new GameObject("TestRoot");

        [TearDown]
        public void TearDown()
        {
            if (_go != null) Object.DestroyImmediate(_go);
        }

        static AstelManifest MakeManifest(
            float mass = 2.5f,
            float[] com = null,
            float friction = 0.4f,
            float restitution = 0.2f
        ) => new AstelManifest
        {
            meters_per_unit = 1.0f,
            l5 = new L5Layer
            {
                mass_props = new MassProps
                {
                    mass_kg = mass,
                    center_of_mass = com ?? new float[] { 0f, 0f, 0f },
                    volume_m3 = 0.01f,
                    inertia_diagonal = new float[] { 0.1f, 0.1f, 0.1f },
                }
            },
            l6 = new L6Layer
            {
                regions = new System.Collections.Generic.List<PhysicsMaterialRegion>
                {
                    new PhysicsMaterialRegion { name = "wood", density_kg_m3 = 700f, friction = friction, restitution = restitution }
                },
                articulation = new System.Collections.Generic.List<ArticulationHint>()
            }
        };

        [Test]
        public void Apply_AddsRigidbody()
        {
            var manifest = MakeManifest();
            new AstelPhysicsSetup(manifest, _go).Apply();
            Assert.IsNotNull(_go.GetComponent<Rigidbody>(), "Rigidbody should be added");
        }

        [Test]
        public void Apply_SetsMass()
        {
            var manifest = MakeManifest(mass: 3.7f);
            new AstelPhysicsSetup(manifest, _go).Apply();
            Assert.AreEqual(3.7f, _go.GetComponent<Rigidbody>().mass, 0.001f);
        }

        [Test]
        public void Apply_SetsCentreOfMass_WithXFlip()
        {
            var manifest = MakeManifest(com: new float[] { 1f, 0.5f, 0.2f });
            new AstelPhysicsSetup(manifest, _go).Apply();
            var com = _go.GetComponent<Rigidbody>().centerOfMass;
            Assert.AreEqual(-1f, com.x, 0.001f, "X should be negated (right→left-hand)");
            Assert.AreEqual(0.5f, com.y, 0.001f);
            Assert.AreEqual(0.2f, com.z, 0.001f);
        }

        [Test]
        public void Apply_UsesDefaultMassIfZero()
        {
            var manifest = MakeManifest(mass: 0f);
            new AstelPhysicsSetup(manifest, _go).Apply();
            Assert.AreEqual(1f, _go.GetComponent<Rigidbody>().mass, 0.001f);
        }

        [Test]
        public void Apply_NoL5_NoRigidbody()
        {
            var manifest = new AstelManifest { l5 = null, l6 = null };
            new AstelPhysicsSetup(manifest, _go).Apply();
            Assert.IsNull(_go.GetComponent<Rigidbody>(), "No Rigidbody without L5");
        }

        [Test]
        public void Apply_NoL6_NoPhysicMaterial_DoesNotThrow()
        {
            var manifest = new AstelManifest
            {
                meters_per_unit = 1f,
                l5 = new L5Layer { mass_props = new MassProps { mass_kg = 1f, center_of_mass = new float[3] } },
                l6 = null
            };
            Assert.DoesNotThrow(() => new AstelPhysicsSetup(manifest, _go).Apply());
        }

        [Test]
        public void Apply_MetersPerUnit_ScalesCOM()
        {
            var manifest = MakeManifest(com: new float[] { 1f, 1f, 1f });
            manifest.meters_per_unit = 0.01f;  // 1 unit = 1 cm
            new AstelPhysicsSetup(manifest, _go).Apply();
            var com = _go.GetComponent<Rigidbody>().centerOfMass;
            Assert.AreEqual(-0.01f, com.x, 0.0001f);
            Assert.AreEqual(0.01f, com.y, 0.0001f);
            Assert.AreEqual(0.01f, com.z, 0.0001f);
        }

        // Parse a REAL engine.json payload (the exact shape produced by
        // astel_gpu.packaging.build_engine_setup, schema astel.engine-setup/v0)
        // through JsonUtility — the contract the producer actually emits, not a
        // hand-built object. Guards against the parser drifting from the emitter.
        const string RealEngineJson = @"{
  ""schema"": ""astel.engine-setup/v0"",
  ""meters_per_unit"": 0.3,
  ""splat_file"": ""l3.spz"",
  ""scale_grounded"": true,
  ""coordinate_system"": {""handedness"": ""right"", ""up_axis"": ""+Y"", ""forward_axis"": ""-Z""},
  ""l5"": {""mass_props"": {""volume_m3"": 0.5, ""mass_kg"": 9.76,
           ""center_of_mass"": [0.1, 0.2, 0.3], ""inertia_diagonal"": [0.4, 0.5, 0.6]}},
  ""l6"": {""regions"": [
            {""name"": ""box"", ""density_kg_m3"": 700.0, ""friction"": 0.5, ""restitution"": 0.2},
            {""name"": ""lid"", ""density_kg_m3"": 7850.0, ""friction"": 0.4, ""restitution"": 0.3}],
           ""articulation"": [{""joint_type"": ""revolute"", ""region_a"": 0, ""region_b"": 1}]},
  ""notes"": []
}";

        [Test]
        public void Parse_RealEngineJson_PopulatesPhysicsFields()
        {
            var m = JsonUtility.FromJson<AstelManifest>(RealEngineJson);
            Assert.AreEqual("astel.engine-setup/v0", m.schema);
            Assert.AreEqual("l3.spz", m.splat_file);
            Assert.AreEqual(0.3f, m.meters_per_unit, 1e-6f);
            Assert.IsTrue(m.scale_grounded);
            Assert.AreEqual(9.76f, m.l5.mass_props.mass_kg, 1e-4f);
            Assert.AreEqual(2, m.l6.regions.Count);
            Assert.AreEqual("box", m.l6.regions[0].name);
            Assert.AreEqual(0.5f, m.l6.regions[0].friction, 1e-6f);
            Assert.AreEqual("revolute", m.l6.articulation[0].joint_type);
            Assert.AreEqual(1, m.l6.articulation[0].region_b);
        }

        [Test]
        public void Parse_RealEngineJson_AppliesToRigidbody()
        {
            var m = JsonUtility.FromJson<AstelManifest>(RealEngineJson);
            new AstelPhysicsSetup(m, _go).Apply();
            var rb = _go.GetComponent<Rigidbody>();
            Assert.IsNotNull(rb);
            Assert.AreEqual(9.76f, rb.mass, 1e-3f);
            // COM X-flipped and scaled by meters_per_unit (0.1 * 0.3, negated).
            Assert.AreEqual(-0.03f, rb.centerOfMass.x, 1e-4f);
        }
    }
}
