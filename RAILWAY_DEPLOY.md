# Railway Deployment Guide

This guide will help you deploy the Clinical Decision Support MVP to Railway.

## Prerequisites

1. A Railway account (sign up at https://railway.app)
2. Your OpenAI API key

## Deployment Steps

### Option 1: Deploy via Railway Web Interface (Recommended)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Configure for Railway deployment"
   git push origin main
   ```

2. **Go to Railway Dashboard**:
   - Visit https://railway.app
   - Sign in with your GitHub account

3. **Create a New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository: `kieran-heidi/hamza-clinical-decision-support-mvp`

4. **Configure Environment Variables**:
   - In your Railway project, go to the "Variables" tab
   - Add the following environment variable:
     - `OPENAI_API_KEY`: Your OpenAI API key (e.g., `sk-...`)

5. **Deploy**:
   - Railway will automatically detect the Dockerfile and start building
   - The deployment will begin automatically
   - Wait for the build to complete (this may take 5-10 minutes)

6. **Generate a Public Domain**:
   - Once deployed, go to the "Settings" tab
   - Under "Networking", click "Generate Domain"
   - Your app will be available at the generated URL

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI**:
   ```bash
   # Using Homebrew (macOS)
   brew install railway
   
   # Or using npm
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize and Deploy**:
   ```bash
   railway init
   railway up
   ```

4. **Set Environment Variables**:
   ```bash
   railway variables set OPENAI_API_KEY=your-api-key-here
   ```

5. **Get your deployment URL**:
   ```bash
   railway domain
   ```

## Important Notes

- The `chroma_db` directory is included in the repository, so the pre-ingested medical guidelines will be available
- Make sure your OpenAI API key is set as an environment variable in Railway
- The first deployment may take longer due to Docker image building
- Railway will automatically rebuild on every push to your main branch (if you enable it in settings)

## Troubleshooting

- **Build fails**: Check the Railway logs in the dashboard
- **App doesn't start**: Verify that `OPENAI_API_KEY` is set correctly
- **Port issues**: Railway automatically sets the `PORT` environment variable, which the Dockerfile now uses

## Cost Considerations

- Railway offers a free tier with limited usage
- Monitor your usage in the Railway dashboard
- Consider upgrading if you need more resources

