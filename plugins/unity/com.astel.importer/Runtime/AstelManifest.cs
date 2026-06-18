// Minimal typed deserialisation of Astel's engine.json sidecar — the flat
// physics-setup descriptor (schema "astel.engine-setup/v0") emitted ALONGSIDE
// the .astel package by the producer (see astel_gpu.packaging.build_engine_setup
// and docs/architecture/coordinate-conventions.md). It denormalises the L5
// collision + L6 material/mass/articulation layers so an engine importer never
// has to walk the nested manifest.json or chase its file-referenced sidecars.
// Only the fields the physics auto-setup needs; unknown fields are ignored.
using System.Collections.Generic;
using UnityEngine;

namespace Astel
{
    [System.Serializable]
    public class AstelManifest
    {
        // Populated by JsonUtility from engine.json
        public string schema;            // "astel.engine-setup/v0"
        public string splat_file;        // the splat to import, e.g. "l3.spz"
        public L5Layer l5;               // null when no watertight surface was derived
        public L6Layer l6;               // null when no physics-material layer exists
        public CoordinateSystem coordinate_system;
        public float meters_per_unit = 1.0f;
        public bool scale_grounded;      // false => mass/length assume 1 unit = 1 m
    }

    [System.Serializable]
    public class CoordinateSystem
    {
        public string handedness;
        public string up_axis;
        public string forward_axis;
        public float meters_per_unit = 1.0f;
    }

    [System.Serializable]
    public class L5Layer
    {
        public MassProps mass_props;
        public string collision_mesh;    // e.g. "l5-collision.obj" (best-effort)
    }

    [System.Serializable]
    public class MassProps
    {
        public float volume_m3;
        public float mass_kg;
        public float[] center_of_mass;   // length 3
        public float[] inertia_diagonal; // length 3 (Ixx, Iyy, Izz in kg·m²)
    }

    [System.Serializable]
    public class L6Layer
    {
        public List<PhysicsMaterialRegion> regions;
        public List<ArticulationHint> articulation;
    }

    [System.Serializable]
    public class PhysicsMaterialRegion
    {
        public string name;
        public float density_kg_m3;
        public float friction;
        public float restitution;
    }

    [System.Serializable]
    public class ArticulationHint
    {
        public string joint_type;  // revolute | prismatic | fixed | free
        public int region_a;
        public int region_b;
        public float[] axis;
    }
}
