import re


def clean_number(text):
    text = re.sub(r'CID:\s*\d+,?', '', text)
    text = text.encode('utf-8').decode('unicode_escape')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(' ,', ',')
    return text


def anonymize_path(file_path: str) -> str:
    patterns = [
        (r'/Users/[^/]+/', '/Users/[USER]/'),
        (r'C:\\Users\\[^\\]+\\', 'C:\\Users\\[USER]\\'),
        (r'/home/[^/]+/', '/home/[USER]/'),
        (r'C:\\Users\\[^\\]+\\Documents\\', 'C:\\Users\\[USER]\\Documents\\'),
        (r'/Users/[^/]+/Documents/', '/Users/[USER]/Documents/'),
    ]
    
    anonymized_path = file_path
    for pattern, replacement in patterns:
        anonymized_path = re.sub(pattern, replacement, anonymized_path)
    
    return anonymized_path


def clean_event(event):
    event = re.sub(r"\s+", " ", event.strip())
    
    if "_" in event:
        event = " ".join(dict.fromkeys(event.replace("_", "").split()))
    return event

def clean_markdown(text)->str:
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    text = re.sub(r'^\s*#{1,6}\s*(.*?)$', r'\1\n', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^\s*>+\s?', '', text, flags=re.MULTILINE)
    text = re.sub(r'</?([a-zA-Z0-9]+)(?:\s[^>]*)?>', lambda m: m.group(0) if m.group(0).startswith('<') and m.group(0).endswith('>') else '', text)
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    return text


def cut_consequence(text)->str:
    text=text.lower()
    patterns = [
        r'resolution:?.*',  
        r'the inner conflict:?.*',
        r'conclusion:?.*',
        r'scene conclusion:?.*',
        r'conclusion of the scene:?.*',
        r'outcome reflection:?.*',
        r'reflection:?.*',
        r'resolution proposal:?.*',
        r'choice and reflection:?.*',
        r'off the scene:?.*',
        r'interlude:?.*',
        r'proposal:?.*',
        r'consequence:?.*',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return text