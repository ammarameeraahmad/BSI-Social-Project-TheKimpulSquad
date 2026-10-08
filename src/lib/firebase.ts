import { getApp, getApps, initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

export function getFirebaseDatabase() {
  const config = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
    databaseURL: process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  };
  const missingConfig = Object.entries(config)
    .filter(([, value]) => !value)
    .map(([key]) => key);

  if (missingConfig.length > 0) {
    throw new Error(`Konfigurasi Firebase belum lengkap: ${missingConfig.join(', ')}`);
  }

  const app = getApps().length > 0
    ? getApp()
    : initializeApp({
      apiKey: config.apiKey!,
      appId: config.appId!,
      databaseURL: config.databaseURL!,
      projectId: config.projectId!,
    });
  return getDatabase(app);
}
