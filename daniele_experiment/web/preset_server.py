#!/usr/bin/env python3
"""
Simple HTTP server to handle preset saving for the Go position labeler.
Run this server alongside the HTML page to enable automatic preset saving.

Usage: python preset_server.py
"""

import json
import yaml
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time


class PresetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/get_presets':
            try:
                # Load current presets from config file
                presets_path = Path(__file__).parent.parent / "configs" / "annotations_presets.yaml"
                presets = self.load_presets_from_config(presets_path)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(presets).encode())
                
            except Exception as e:
                print(f"Error handling get_presets request: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        if self.path == '/save_preset':
            try:
                # Read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract preset data
                preset_name = data.get('name', '').strip()
                global_labels = data.get('global_labels', {})
                move_labels = data.get('move_labels', {})
                
                if not preset_name:
                    self.send_error(400, "Preset name is required")
                    return
                
                # Save to config file
                presets_path = Path(__file__).parent.parent / "configs" / "annotations_presets.yaml"
                success = self.save_preset_to_config(presets_path, preset_name, global_labels, move_labels)
                
                if success:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'message': f'Preset "{preset_name}" saved'}).encode())
                else:
                    self.send_error(500, "Failed to save preset")
                    
            except Exception as e:
                print(f"Error handling save_preset request: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")
    
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def load_presets_from_config(self, presets_path: Path) -> dict:
        """Load presets from YAML config file and convert to JavaScript format."""
        if not presets_path.exists():
            return {}
        
        try:
            with presets_path.open("r", encoding="utf-8") as f:
                presets_data = yaml.safe_load(f) or {}
            
            # Convert YAML structure to JavaScript format
            js_presets = {}
            for preset_name, preset_config in presets_data.items():
                if isinstance(preset_config, dict) and 'global_labels' in preset_config and 'move_labels' in preset_config:
                    js_presets[preset_name] = {
                        'globalLabels': preset_config.get('global_labels', {}),
                        'moveLabels': preset_config.get('move_labels', {})
                    }
            
            return js_presets
        except Exception as e:
            print(f"Warning: Could not load presets from {presets_path}: {e}")
            return {}
    
    def save_preset_to_config(self, presets_path: Path, preset_name: str, global_labels: dict, move_labels: dict) -> bool:
        """Save a new preset to the YAML config file."""
        try:
            # Ensure directory exists
            presets_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing presets
            if presets_path.exists():
                with presets_path.open("r", encoding="utf-8") as f:
                    existing_presets = yaml.safe_load(f) or {}
            else:
                existing_presets = {}
            
            # Add the new preset
            existing_presets[preset_name] = {
                'description': f'User-defined preset: {preset_name}',
                'global_labels': global_labels,
                'move_labels': move_labels
            }
            
            # Save back to file
            with presets_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(existing_presets, f, default_flow_style=False, sort_keys=False)
            
            print(f"Successfully saved preset '{preset_name}' to {presets_path}")
            return True
            
        except Exception as e:
            print(f"Error saving preset to {presets_path}: {e}")
            return False
    
    def log_message(self, format, *args):
        # Override to reduce log noise
        if "save_preset" in format % args:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")


def start_preset_server(port=8001):
    """Start the preset server in a separate thread."""
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, PresetHandler)
    
    print(f"Starting preset server on http://localhost:{port}")
    print("This server handles automatic preset saving for the Go position labeler.")
    print("Press Ctrl+C to stop.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down preset server...")
        httpd.shutdown()


if __name__ == "__main__":
    start_preset_server()
