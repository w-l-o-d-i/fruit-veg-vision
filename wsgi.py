import os
from backend import app, BASE_DIR

def application(environ, start_response):
    os.environ['ENV'] = environ.get('ENV', 'production')
    path = environ.get('PATH_INFO', '')
    
    if path.startswith('/static/'):
        static_file = os.path.join(BASE_DIR, 'lovable', 'static', path[8:])
        if os.path.exists(static_file):
            with open(static_file, 'rb') as f:
                data = f.read()
            content_type = 'application/octet-stream'
            if path.endswith('.js'):
                content_type = 'application/javascript'
            elif path.endswith('.css'):
                content_type = 'text/css'
            elif path.endswith('.png'):
                content_type = 'image/png'
            elif path.endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            start_response('200 OK', [('Content-Type', content_type)])
            return [data]
    
    return app(environ, start_response)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend:app', host='0.0.0.0', port=8000, reload=True)
