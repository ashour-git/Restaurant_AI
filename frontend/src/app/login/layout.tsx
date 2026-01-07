import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Login - RestAI',
  description: 'Sign in to your RestAI account',
};

export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
