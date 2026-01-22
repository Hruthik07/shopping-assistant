# Frontend - AI Shopping Assistant

A modern, responsive web interface for the AI Shopping Assistant.

## 🚀 Features

- **Real-time Chat Interface**: Interactive chat with the AI assistant
- **Product Display**: Beautiful product cards with details
- **Session Management**: Persistent session tracking
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Clean, gradient-based design

## 📁 Files

- `index.html` - Main HTML structure
- `styles.css` - Styling and responsive design
- `app.js` - JavaScript functionality and API integration

## 🛠️ Setup

### Option 1: Simple HTTP Server (Recommended)

```bash
# Python 3
cd frontend
python -m http.server 8080

# Or Node.js
npx http-server -p 8080
```

Then open: `http://localhost:8080`

### Option 2: Serve with Backend

Add static file serving to FastAPI:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

## ⚙️ Configuration

Edit `app.js` to change API URL:

```javascript
const API_BASE_URL = 'http://localhost:3565';  // Change this
```

## 🎨 Customization

- **Colors**: Edit gradient colors in `styles.css`
- **Layout**: Modify grid layout in `styles.css`
- **Features**: Add new functionality in `app.js`

## 📱 Responsive Design

- Desktop: Side-by-side chat and products
- Tablet: Stacked layout
- Mobile: Full-width chat, collapsible products

## 🔗 API Integration

The frontend connects to:
- `POST /api/chat/` - Send messages
- `GET /health` - Health check

## 🚀 Next Steps

1. Add WebSocket support for streaming responses
2. Add cart functionality
3. Add product image display
4. Add search filters
5. Add user authentication


