#pragma once

#include "Modules/ModuleManager.h"

class FAstelPluginModule : public IModuleInterface
{
public:
    void StartupModule() override;
    void ShutdownModule() override;
};
