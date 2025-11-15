#!/bin/bash

echo "🚀 Setting up MediChain-FL Frontend..."
echo "======================================"

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ -d "node_modules" ]; then
    echo "✅ node_modules already exists, skipping install"
else
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p src/components
mkdir -p src/hooks
mkdir -p public

# Check if components exist
if [ ! -f "src/components/LoginScreen.js" ]; then
    echo "⚠️  Warning: Component files not found!"
    echo "   Please copy the component files from the artifacts"
fi

echo ""
echo "✅ Frontend setup complete!"
echo ""
echo "To start the development server:"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "The dashboard will be available at http://localhost:3000"