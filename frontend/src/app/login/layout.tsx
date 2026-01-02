import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Login - RestaurantAI',
  description: 'Sign in to your RestaurantAI account',
};

export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
