# Deployment & Operational Guide: Breast DCE-MRI Research Prototype

This document outlines the deployment and execution architecture for the **3D DCE-MRI pCR research prototype**. It is not a validated diagnostic or treatment system and does not estimate tumor volume.

---

## 1. System Architecture & Topology

The platform is designed as a modular, decoupled full-stack medical AI architecture:

```
┌────────────────────────────────────────────────────────┐
│                   React.js Frontend                    │
│      (Vite + Tailwind CSS + Lucide + Chart.js)        │
│          Port: 5173 (Dev) / Port 80 (Nginx Prod)       │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / REST / JWT
                            ▼
┌────────────────────────────────────────────────────────┐
│               Node.js + Express Backend                │
│ (Auth, RBAC, Multer NIfTI IO, PDFKit, Audit Compliance)│
│                      Port: 5000                        │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              │ Internal API              │ Mongoose ORM
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│ Python FastAPI Microservice│ │   MongoDB / Atlas DB     │
│(PyTorch + Grad-CAM + 3D)  │ │ (Collections: Users,     │
│       Port: 8000          │ │  Patients, Predictions,  │
│  Loads checkpoints/pt     │ │  Reports, AuditLogs)     │
└───────────────────────────┘ └──────────────────────────┘
```

---

## 2. Quick Start: Local Development

### Prerequisites
- Node.js v18+ and npm
- Python 3.10+ (with PyTorch, MONAI, NiBabel installed in `.venv`)
- (Optional) MongoDB running locally or on MongoDB Atlas

### Step 1: Start the Python FastAPI AI Microservice
```bash
# From project root
.venv\Scripts\activate
uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Health verification: `curl http://localhost:8000/health`. The service requires a loadable classifier checkpoint; otherwise it reports degraded and rejects inference.*

For browser deployments, set `CORS_ORIGINS` in the AI service to a comma-separated list of exact front-end origins (for example `http://localhost:5173`).

### Step 2: Start the Node.js Express Backend
```bash
cd server
npm install
npm run dev # or node src/server.js
```
*API Base: `http://localhost:5000/api`*
*(Note: If MongoDB is offline, the backend activates a high-performance in-memory data store automatically).*

### Step 3: Start the React Frontend
```bash
cd client
npm install
npm run dev
```
*Frontend URL: `http://localhost:5173`*

---

## 3. Synthetic Demo Credentials (Local Walkthrough Only)

These accounts and patient profiles are synthetic and exist only for a local interface walkthrough. Do not enter real patient information. The app has no verified clinical performance or compliance status.

| Persona | Email | Password | Assigned Permissions |
| :--- | :--- | :--- | :--- |
| **Demo doctor account** | `doctor@hospital.org` | `Password123!` | Local UI walkthrough |
| **Demo researcher account** | `researcher@oncology.edu` | `Password123!` | Local UI walkthrough |
| **Demo administrator account** | `admin@cancercenter.org` | `Password123!` | Local UI walkthrough |

---

## 4. Single-Command Docker Compose Deployment

To build and run all 4 containers together (MongoDB, AI Microservice, Backend API, React Frontend):

```bash
docker-compose up --build -d
```

Check running services:
```bash
docker-compose ps
```

Access points:
- **Web Application**: `http://localhost:80`
- **Backend API**: `http://localhost:5000/api`
- **AI Microservice**: `http://localhost:8000/docs`
- **MongoDB**: `localhost:27017`

---

## 5. Cloud Production Deployment

### 5.1 MongoDB Atlas Cloud Database Setup
1. Create a free M0 cluster on [MongoDB Atlas](https://www.mongodb.com/atlas).
2. Under **Network Access**, add IP `0.0.0.0/0` (or your cloud host IP).
3. Under **Database Access**, create a user (e.g. `cancer_admin`).
4. Copy the connection URI:
   ```env
   MONGODB_URI=mongodb+srv://cancer_admin:<PASSWORD>@cluster0.mongodb.net/breast_cancer_db?retryWrites=true&w=majority
   ```

### 5.2 Backend API Deployment on Render
1. Push this repository to GitHub.
2. In Render, select **New > Web Service** and connect the repository.
3. Configure settings:
   - **Root Directory**: `server`
   - **Environment**: `Node`
   - **Build Command**: `npm install`
   - **Start Command**: `node src/server.js`
4. Add Environment Variables:
   - `PORT`: `5000`
   - `NODE_ENV`: `production`
   - `MONGODB_URI`: `<Your MongoDB Atlas URI>`
   - `JWT_SECRET`: `<Your Secure JWT Secret>`
   - `AI_SERVICE_URL`: `<Your Deployed AI Service URL>`

### 5.3 AI Microservice on Docker / Render / Hugging Face Spaces
1. Deploy via Docker using `ai_service/Dockerfile`.
2. Ensure `checkpoints/best_3d_classifier.pt` is committed or mounted as a cloud volume.
3. Expose port `8000`.

### 5.4 Frontend Deployment on Vercel
1. Import repository in [Vercel](https://vercel.com).
2. Set **Root Directory** to `client`.
3. Framework Preset: `Vite`.
4. Build Command: `npm run build`.
5. Output Directory: `dist`.
6. Add Environment Variable:
   - `VITE_API_URL`: `https://your-render-backend.onrender.com/api`
7. Click **Deploy**.

---

## 6. Testing & Demonstration Workflow

1. Review the research-only limitations on the landing page.
2. Sign in with a local demo account if the database was seeded.
3. Upload a supported 3D subtraction NIfTI or 4D DCE NIfTI containing pre- and post-contrast phases. Synthetic examples are illustrative only.
4. Review the pCR estimate and optional attribution visualization. Do not interpret attribution as a lesion contour or clinical finding.
5. Analytics remain unavailable until a valid, patient-held-out evaluation report is explicitly configured.
6. Generated PDFs are research summaries, not diagnostic reports.
