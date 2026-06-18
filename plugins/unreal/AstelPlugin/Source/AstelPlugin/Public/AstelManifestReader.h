#pragma once

#include "CoreMinimal.h"
#include "AstelManifestReader.generated.h"

/**
 * Mirrors Astel's engine.json sidecar (schema "astel.engine-setup/v0") — the
 * flat physics-setup descriptor emitted ALONGSIDE the .astel package by the
 * producer (astel_gpu.packaging.build_engine_setup). It denormalises the L5
 * collision + L6 material/mass/articulation layers so this plugin never has to
 * walk the nested manifest.json or chase its file-referenced sidecars.
 * Only the L5/L6 physics fields consumed by the auto-setup are mapped.
 */

USTRUCT(BlueprintType)
struct FAstelMassProps
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float VolumeM3 = 0.f;

    UPROPERTY(BlueprintReadOnly)
    float MassKg = 1.f;

    UPROPERTY(BlueprintReadOnly)
    FVector CenterOfMassM = FVector::ZeroVector;

    UPROPERTY(BlueprintReadOnly)
    FVector InertiaDiagonal = FVector::OneVector;
};

USTRUCT(BlueprintType)
struct FAstelPhysicsRegion
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString Name;

    UPROPERTY(BlueprintReadOnly)
    float DensityKgM3 = 1000.f;

    UPROPERTY(BlueprintReadOnly)
    float Friction = 0.6f;

    UPROPERTY(BlueprintReadOnly)
    float Restitution = 0.1f;
};

USTRUCT(BlueprintType)
struct FAstelArticulationHint
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString JointType;  // revolute | prismatic | fixed | free

    UPROPERTY(BlueprintReadOnly)
    int32 RegionA = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 RegionB = 1;

    UPROPERTY(BlueprintReadOnly)
    FVector Axis = FVector::ForwardVector;
};

USTRUCT(BlueprintType)
struct FAstelManifest
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float MetersPerUnit = 1.f;

    UPROPERTY(BlueprintReadOnly)
    FAstelMassProps MassProps;

    UPROPERTY(BlueprintReadOnly)
    TArray<FAstelPhysicsRegion> Regions;

    UPROPERTY(BlueprintReadOnly)
    TArray<FAstelArticulationHint> ArticulationHints;

    bool IsValid() const { return MassProps.MassKg > 0.f; }
};

/**
 * Parses engine.json and auto-configures an Actor's physics.
 *
 * engine.json is a sibling DELIVERY artifact — download it next to the splat
 * file it names; it is NOT a member inside the package.astel zip.
 *
 * Coordinate convention:
 *   data is in 3DGS world (right-handed, +Y up, metres).
 *   UE5 is left-handed, +Z up, +X forward, centimetres.
 *   Transform: pos_ue = (−z×100, x×100, y×100)  [cm].
 *   See docs/architecture/coordinate-conventions.md.
 */
UCLASS()
class ASTELPLUGIN_API UAstelManifestReader : public UObject
{
    GENERATED_BODY()

public:
    /** Parse an engine.json string into FAstelManifest. */
    static FAstelManifest ParseManifestJson(const FString& JsonString);

    /** Load engine.json from a folder of downloaded artifacts. */
    static FAstelManifest LoadFromDirectory(const FString& ArtifactDir);

    /**
     * Apply physics configuration to a static mesh component.
     *
     * Sets:
     *   - Mass override (FBodyInstance.bOverrideMass = true, MassInKgOverride).
     *   - Centre of mass (COMNudge in UE5 local UU coords).
     *   - Physical material friction + restitution (created at runtime).
     *   Collision hulls: log a hint — use L5 hull OBJs with UBodySetup manually
     *   or via the AstelCollisionBuilder blueprint utility.
     */
    static void ApplyPhysics(
        UStaticMeshComponent* MeshComp,
        const FAstelManifest& Manifest
    );

private:
    /** Convert a 3DGS-world position (metres) to UE5 local units (cm). */
    static FVector ConvertPosition(float X, float Y, float Z, float MetersPerUnit);
};
