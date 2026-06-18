#include "AstelManifestReader.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "PhysicsEngine/BodyInstance.h"
#include "PhysicalMaterials/PhysicalMaterial.h"
#include "Components/StaticMeshComponent.h"

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

static float GetFloat(
    const TSharedPtr<FJsonObject>& Obj, const FString& Key, float Default = 0.f)
{
    double Val = 0.0;
    return Obj->TryGetNumberField(Key, Val) ? (float)Val : Default;
}

static FVector GetVec3(
    const TSharedPtr<FJsonObject>& Obj, const FString& Key, FVector Default = FVector::ZeroVector)
{
    const TArray<TSharedPtr<FJsonValue>>* Arr;
    if (!Obj->TryGetArrayField(Key, Arr) || Arr->Num() < 3)
        return Default;
    return FVector(
        (float)(*Arr)[0]->AsNumber(),
        (float)(*Arr)[1]->AsNumber(),
        (float)(*Arr)[2]->AsNumber()
    );
}

// ---------------------------------------------------------------------------
// UAstelManifestReader — public API
// ---------------------------------------------------------------------------

FAstelManifest UAstelManifestReader::ParseManifestJson(const FString& JsonString)
{
    FAstelManifest Result;

    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
        return Result;

    Result.MetersPerUnit = GetFloat(Root, TEXT("meters_per_unit"), 1.f);

    // L5 mass_props
    const TSharedPtr<FJsonObject>* L5Obj;
    if (Root->TryGetObjectField(TEXT("l5"), L5Obj))
    {
        const TSharedPtr<FJsonObject>* MpObj;
        if ((*L5Obj)->TryGetObjectField(TEXT("mass_props"), MpObj))
        {
            Result.MassProps.VolumeM3  = GetFloat(*MpObj, TEXT("volume_m3"));
            Result.MassProps.MassKg    = GetFloat(*MpObj, TEXT("mass_kg"), 1.f);
            FVector com = GetVec3(*MpObj, TEXT("center_of_mass"));
            Result.MassProps.CenterOfMassM = com;
            FVector inertia = GetVec3(*MpObj, TEXT("inertia_diagonal"), FVector::OneVector);
            Result.MassProps.InertiaDiagonal = inertia;
        }
    }

    // L6 regions
    const TSharedPtr<FJsonObject>* L6Obj;
    if (Root->TryGetObjectField(TEXT("l6"), L6Obj))
    {
        const TArray<TSharedPtr<FJsonValue>>* RegArr;
        if ((*L6Obj)->TryGetArrayField(TEXT("regions"), RegArr))
        {
            for (auto& Val : *RegArr)
            {
                auto Reg = Val->AsObject();
                FAstelPhysicsRegion R;
                R.Name         = Reg->GetStringField(TEXT("name"));
                R.DensityKgM3  = GetFloat(Reg, TEXT("density_kg_m3"), 1000.f);
                R.Friction     = GetFloat(Reg, TEXT("friction"),    0.6f);
                R.Restitution  = GetFloat(Reg, TEXT("restitution"), 0.1f);
                Result.Regions.Add(R);
            }
        }

        // Articulation hints
        const TArray<TSharedPtr<FJsonValue>>* ArtArr;
        if ((*L6Obj)->TryGetArrayField(TEXT("articulation"), ArtArr))
        {
            for (auto& Val : *ArtArr)
            {
                auto Hint = Val->AsObject();
                FAstelArticulationHint H;
                H.JointType = Hint->GetStringField(TEXT("joint_type"));
                int32 RA = 0, RB = 1;
                Hint->TryGetNumberField(TEXT("region_a"), RA);
                Hint->TryGetNumberField(TEXT("region_b"), RB);
                H.RegionA = RA;
                H.RegionB = RB;
                H.Axis = GetVec3(Hint, TEXT("axis"), FVector::ForwardVector);
                Result.ArticulationHints.Add(H);
            }
        }
    }

    return Result;
}

FAstelManifest UAstelManifestReader::LoadFromDirectory(const FString& ArtifactDir)
{
    // engine.json is a sibling delivery artifact (download it next to the splat
    // it names); it is not a member inside the package.astel zip.
    FString EnginePath = FPaths::Combine(ArtifactDir, TEXT("engine.json"));
    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *EnginePath))
    {
        UE_LOG(LogTemp, Error, TEXT("AstelPlugin: could not read %s"), *EnginePath);
        return FAstelManifest{};
    }
    return ParseManifestJson(JsonString);
}

FVector UAstelManifestReader::ConvertPosition(
    float X, float Y, float Z, float MetersPerUnit)
{
    // 3DGS world (right-hand, +Y up, metres) → UE5 (left-hand, +Z up, cm).
    // Axis remap: (x,y,z) → (-z*100, x*100, y*100)  [cm in UE units].
    float cm = 100.f * MetersPerUnit;
    return FVector(-Z * cm, X * cm, Y * cm);
}

void UAstelManifestReader::ApplyPhysics(
    UStaticMeshComponent* MeshComp,
    const FAstelManifest& Manifest)
{
    if (!MeshComp) return;

    // --- Mass override ---
    FBodyInstance& BI = MeshComp->BodyInstance;
    if (Manifest.MassProps.MassKg > 0.f)
    {
        BI.bOverrideMass = true;
        BI.SetMassOverride(Manifest.MassProps.MassKg);
    }

    // --- Centre of mass (COMNudge in component local space, cm) ---
    const FVector& com = Manifest.MassProps.CenterOfMassM;
    BI.COMNudge = ConvertPosition(com.X, com.Y, com.Z, Manifest.MetersPerUnit);

    // --- Physical material (from region 0) ---
    if (Manifest.Regions.Num() > 0)
    {
        const FAstelPhysicsRegion& R = Manifest.Regions[0];
        UPhysicalMaterial* Mat = NewObject<UPhysicalMaterial>(
            MeshComp, FName(TEXT("AstelPhysMat_") + R.Name));
        Mat->Friction    = R.Friction;
        Mat->Restitution = R.Restitution;
        BI.PhysMaterialOverride = Mat;
    }

    // --- Articulation hints ---
    for (const FAstelArticulationHint& H : Manifest.ArticulationHints)
    {
        UE_LOG(LogTemp, Log,
            TEXT("AstelPlugin: articulation hint: regions %d<->%d joint=%s"),
            H.RegionA, H.RegionB, *H.JointType);
    }

    MeshComp->RecreatePhysicsState();
}
