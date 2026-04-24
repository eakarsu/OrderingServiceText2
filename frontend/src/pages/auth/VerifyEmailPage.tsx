import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../../api/client';

export default function VerifyEmailPage() {
  const { token } = useParams();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');

  useEffect(() => {
    api.get(`/auth/verify-email/${token}`)
      .then(() => setStatus('success'))
      .catch(() => setStatus('error'));
  }, [token]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-8 text-center">
        {status === 'loading' && <p className="text-gray-600">Verifying...</p>}
        {status === 'success' && (
          <>
            <h1 className="text-2xl font-bold text-green-600 mb-2">Email Verified!</h1>
            <Link to="/login" className="text-blue-600 hover:underline">Go to login</Link>
          </>
        )}
        {status === 'error' && (
          <>
            <h1 className="text-2xl font-bold text-red-600 mb-2">Verification Failed</h1>
            <p className="text-gray-600">The link may be expired or invalid.</p>
          </>
        )}
      </div>
    </div>
  );
}
