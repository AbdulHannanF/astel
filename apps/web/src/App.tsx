import { BrowserRouter, Route, Routes } from "react-router-dom";

import { ErrorBoundary } from "./components/ErrorBoundary.tsx";
import { SiteLayout } from "./components/site/SiteLayout.tsx";
import { DocsPage } from "./pages/DocsPage.tsx";
import { FeaturesPage } from "./pages/FeaturesPage.tsx";
import { LayerInspectorFeature } from "./pages/features/LayerInspectorFeature.tsx";
import { PhysicsSandboxFeature } from "./pages/features/PhysicsSandboxFeature.tsx";
import { RelightStudioFeature } from "./pages/features/RelightStudioFeature.tsx";
import { TruthMeterFeature } from "./pages/features/TruthMeterFeature.tsx";
import { GalleryAssetPage } from "./pages/GalleryAssetPage.tsx";
import { GalleryPage } from "./pages/GalleryPage.tsx";
import { HomePage } from "./pages/HomePage.tsx";
import { NotFoundPage } from "./pages/NotFoundPage.tsx";
import { PipelinePage } from "./pages/PipelinePage.tsx";
import { PricingPage } from "./pages/PricingPage.tsx";
import { SelfHostPage } from "./pages/SelfHostPage.tsx";
import { StudioPage } from "./pages/StudioPage.tsx";

/** The route tree, sans Router, so tests can mount it under a MemoryRouter. */
export function AppRoutes(): React.JSX.Element {
  return (
    <Routes>
      {/* Studio is full-bleed, outside the marketing shell. */}
      <Route path="/studio" element={<StudioPage />} />
      {/* Marketing shell (header + footer) wraps everything else. */}
      <Route element={<SiteLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/features" element={<FeaturesPage />} />
        <Route path="/features/layer-inspector" element={<LayerInspectorFeature />} />
        <Route path="/features/truth-meter" element={<TruthMeterFeature />} />
        <Route path="/features/physics-sandbox" element={<PhysicsSandboxFeature />} />
        <Route path="/features/relight-studio" element={<RelightStudioFeature />} />
        <Route path="/how-it-works" element={<PipelinePage />} />
        <Route path="/gallery" element={<GalleryPage />} />
        <Route path="/gallery/:id" element={<GalleryAssetPage />} />
        <Route path="/pricing" element={<PricingPage />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/self-host" element={<SelfHostPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

export function App(): React.JSX.Element {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </ErrorBoundary>
  );
}
