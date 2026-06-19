# GPT Engineer Web UI

A web-based interface for GPT Engineer that provides the same functionality as the CLI but through a modern web browser interface.

## Features

- **Multi-turn conversations**: Engage in continuous conversations with the AI
- **Dynamic mode selection**: Automatically determines the appropriate mode (generate, improve, debug, misc) based on your input
- **Real-time preview**: See generated files and code in real-time
- **Session management**: Maintain conversation history across browser sessions
- **Modern UI**: Clean, dark-themed interface inspired by modern development tools

## Installation

1. Install the required dependencies:
```bash
poetry install
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key
```

## Usage

### Starting the Web Server

You can start the web server in several ways:

#### Option 1: Using the CLI script
```bash
gpte-web
```

#### Option 2: Using Python module
```bash
python -m gpt_engineer.applications.web
```

#### Option 3: Direct Flask app
```bash
python gpt_engineer/applications/web/app.py
```

### Command Line Options

```bash
gpte-web --help
```

Available options:
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 5001)
- `--debug`: Enable debug mode
- `--reload`: Enable auto-reload on code changes

### Examples

```bash
# Start with default settings
gpte-web

# Start on a different port
gpte-web --port 8080

# Start with debug mode and auto-reload
gpte-web --debug --reload

# Start on all interfaces
gpte-web --host 0.0.0.0 --port 8080
```

## Web Interface

Once the server is running, open your browser and navigate to:
```
http://localhost:5001
```

### Interface Components

1. **Left Panel - Chat Area**
   - Input field for prompts
   - Conversation history
   - Action buttons for attachments, links, images, and new files

2. **Right Panel - Preview Area**
   - Code and Preview tabs
   - Real-time display of generated files
   - File tree view
   - Code syntax highlighting

3. **Sidebar**
   - Quick access icons
   - Session management

### Using the Interface

1. **Start a conversation**: Type your prompt in the input field and press Enter or click the send button
2. **View results**: Generated files will appear in the preview panel
3. **Continue the conversation**: Ask follow-up questions to improve or modify the generated code
4. **Switch between views**: Use the Code and Preview tabs to see different representations

## API Endpoints

The web application provides several REST API endpoints:

- `POST /api/chat`: Send a prompt and get a response
- `GET /api/session/<session_id>/history`: Get conversation history
- `GET /api/session/<session_id>/files`: Get current files for a session
- `GET /api/sessions`: List all active sessions
- `DELETE /api/session/<session_id>`: Delete a session

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MODEL_NAME`: Model to use (default: gpt-4o)
- `TEMPERATURE`: Temperature for AI responses (default: 0.1)
- `AZURE_ENDPOINT`: Azure OpenAI endpoint (optional)
- `SECRET_KEY`: Flask secret key (default: dev-secret-key-change-in-production)

### Project Structure

```
gpt_engineer/applications/web/
├── __init__.py
├── app.py              # Main Flask application
├── __main__.py         # CLI entry point
├── templates/
│   └── index.html      # Web interface template
└── README.md           # This file
```

## Development

### Running in Development Mode

```bash
# Install dependencies
poetry install

# Set up environment
export OPENAI_API_KEY="your-key"

# Run with debug and auto-reload
gpte-web --debug --reload
```

### Project Files

Generated projects are stored in the `projects/` directory with session-specific folders:
```
projects/
└── web_session_<session_id>/
    ├── generated_files.py
    ├── README.md
    └── ...
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Make sure you've set the environment variable or created a `.env` file

2. **"Module not found" errors**
   - Ensure you're running from the project root directory
   - Try installing dependencies with `poetry install`

3. **Port already in use**
   - Use a different port with `--port 8080`
   - Check if another process is using the port

4. **CORS errors**
   - The application includes CORS support, but if you're accessing from a different domain, you may need to configure it

### Debug Mode

Run with debug mode to see detailed error messages:
```bash
gpte-web --debug
```

## Contributing

To contribute to the web interface:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the main project LICENSE file for details.
