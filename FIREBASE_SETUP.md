# Firebase Realtime Database

Gesture samples are stored at `/samples` in Firebase Realtime Database. The
database is intentionally shared and unauthenticated: anyone who can reach the
deployed app can read, add, edit, or delete samples. Do not store sensitive
information in this database. Public write access can also be abused and may
incur Firebase charges.

## Firebase project setup

1. Create a Firebase project and register a Web app in the Firebase console.
2. Create a Realtime Database and copy its database URL.
3. In **Realtime Database → Rules**, publish the contents of
   [`database.rules.json`](./database.rules.json). This grants public read/write
   access only under `/samples`.
4. Copy `.env.example` to `.env.local` for local development and replace each
   placeholder with the matching value from the Firebase Web app settings.
5. Add the same four variables to the Vercel project's Environment Variables
   for each environment you deploy, then redeploy.

The Firebase Web API key and app ID are client-side configuration, not access
controls. The Realtime Database rules determine who can access the data.

Existing samples in this browser's IndexedDB are copied to Firebase the first
time the app connects successfully. They are left in IndexedDB as a local
backup. Samples are shared live between clients. Trained TensorFlow models are
still stored locally in each browser and must be trained separately there.

To publish the rules with the Firebase CLI instead of the console, run
`firebase deploy --only database` after configuring the Firebase project.
