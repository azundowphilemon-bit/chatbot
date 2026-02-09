import re

TUTORIAL_FILE = "documents/w3schools_python.md"

def get_topics():
    """Parses the tutorial file to return a list of topics (H1 headers)."""
    topics = []
    try:
        with open(TUTORIAL_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            # We treat H1 headers (# Header) as main topics
            # Skipping the first general header if it's just metadata title
            if line.startswith("# ") and "W3Schools Python Tutorial" not in line:
                topic = line.strip("# ").strip()
                topics.append(topic)
                
        return topics
    except FileNotFoundError:
        return ["Introduction"] # Fallback

def get_topic_content(topic_name):
    """Returns the content for a specific topic."""
    content = ""
    is_capturing = False
    
    try:
        with open(TUTORIAL_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith("# "):
                current_topic = line.strip("# ").strip()
                if current_topic == topic_name:
                    is_capturing = True
                elif is_capturing:
                    # Stop capturing at the next H1
                    break
            
            if is_capturing:
                content += line
                
        return content
    except:
        return ""
