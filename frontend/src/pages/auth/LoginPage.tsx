import { useRef as useDemoLoginRef, useState as useDemoLoginState } from 'react';
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import toast from 'react-hot-toast';

export default function LoginPage() {
  const demoLoginLock = useDemoLoginRef(false);
  const [demoLoginBusy, setDemoLoginBusy] = useDemoLoginState(false);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent, demoLoginCredentials: { email: string; password: string } | undefined = undefined) => {
    e.preventDefault();
    setLoading(true);
    try {
      await login((demoLoginCredentials?.email ?? email), (demoLoginCredentials?.password ?? password));
      toast.success('Welcome back!');
      navigate('/dashboard');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-8">
        <h1 className="text-2xl font-bold text-gray-900 text-center mb-6">Sign In</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password" required value={password} onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            type="button"
            onClick={async (event) => {
 if (demoLoginLock.current) return;
 demoLoginLock.current = true; setDemoLoginBusy(true);
 const values: {email?: string; password?: string; error?: unknown} = {};
 const captureDemoValue = (key: "email" | "password" | "error", value: any) => { values[key] = value; return value; };
 try {
  await (async () => { setEmail(captureDemoValue("email", import.meta.env.VITE_DEMO_EMAIL || '')); setPassword(captureDemoValue("password", import.meta.env.VITE_DEMO_PASSWORD || '')); })();
  if (values.error || typeof values.email !== 'string' || !values.email || typeof values.password !== 'string' || !values.password) {  return; }
  await handleSubmit(event, { email: values.email, password: values.password });
 } finally { demoLoginLock.current = false; setDemoLoginBusy(false); }
}}
            disabled={demoLoginBusy || (!import.meta.env.VITE_DEMO_EMAIL || !import.meta.env.VITE_DEMO_PASSWORD)}
            aria-label="Log In as Demo"
            style={{ width: '100%', marginBottom: '12px', padding: '10px 14px', borderRadius: '8px', border: '1px solid currentColor', background: 'transparent', cursor: 'pointer' }}
          >
            Log In as Demo
          </button>
          <button
            type="submit" disabled={loading}
            className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <div className="mt-4 text-center text-sm text-gray-600 space-y-2">
          <Link to="/forgot-password" className="text-blue-600 hover:underline block">Forgot password?</Link>
          <span>Don't have an account? <Link to="/register" className="text-blue-600 hover:underline">Sign up</Link></span>
        </div>
      </div>
    </div>
  );
}
