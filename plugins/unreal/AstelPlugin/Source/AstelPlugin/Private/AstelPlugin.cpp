#include "AstelPlugin.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FAstelPluginModule, AstelPlugin)

void FAstelPluginModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("AstelPlugin: module loaded"));
}

void FAstelPluginModule::ShutdownModule() {}
