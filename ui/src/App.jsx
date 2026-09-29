import { NavLink, Route, Routes, Navigate } from 'react-router-dom';
import { SignedIn, SignedOut, SignIn, UserButton } from '@clerk/clerk-react';
import Dashboard from './pages/Dashboard.jsx';
import Extractor from './pages/Extractor.jsx';
import DocumentDetail from './pages/DocumentDetail.jsx';
import Templates from './pages/Templates.jsx';
import TemplateDetail from './pages/TemplateDetail.jsx';

const navClass = ({ isActive }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
    isActive ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100'
  }`;

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-800">
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 h-14">
          <span className="font-bold text-gray-900">Form Extract</span>
          <div className="flex items-center gap-4">
            <SignedIn>
              <div className="flex items-center gap-1">
                <NavLink to="/" end className={navClass}>
                  Dashboard
                </NavLink>
                <NavLink to="/extract" className={navClass}>
                  New Extraction
                </NavLink>
                <NavLink to="/templates" className={navClass}>
                  Templates
                </NavLink>
              </div>
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </nav>

      <Routes>
        <Route path="/login" element={
          <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center p-6">
            <SignIn redirectUrl="/" />
          </div>
        } />
        <Route path="/" element={<Dashboard />} />
        <Route path="/extract" element={<Extractor />} />
        <Route path="/document/:id" element={<DocumentDetail />} />
        <Route path="/templates" element={<Templates />} />
        <Route path="/templates/:id" element={<TemplateDetail />} />
        <Route
          path="*"
          element={
            <SignedIn>
              <div className="max-w-7xl mx-auto p-6">
                <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-10 text-center">
                  <p className="text-gray-700 font-medium">Page not found.</p>
                  <NavLink
                    to="/"
                    className="inline-block mt-3 text-sm font-semibold text-blue-600 hover:text-blue-800"
                  >
                    ← Back to dashboard
                  </NavLink>
                </div>
              </div>
            </SignedIn>
          }
        />
      </Routes>
      <SignedOut>
        <Routes>
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </SignedOut>
    </div>
  );
}