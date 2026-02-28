from google import genai
from google.adk.agents import LlmAgent
import os

# Initialize google-genai Client
client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
)

# No tools are needed for this agent
tools = []

print("Initializing Story Agent...")

story_agent = LlmAgent(
    name="story_agent",
    model="gemini-1.5-flash",
    client=client,
    instructions="""You are a creative assistant for a children's storybook app.
Your task is to generate a short, charming story based on user-provided keywords.
The story must be structured into exactly 4 scenes and formatted as a valid JSON object.

**Story Structure Requirements:**
1.  **Total Word Count:** 100-200 words.
2.  **Language:** Simple, engaging, and suitable for all audiences.
3.  **Scenes:** Exactly 4 scenes, following this narrative arc:
    -   Scene 1: **The Setup** (Introduce characters and setting)
    -   Scene 2: **The Inciting Incident** (The main conflict or event begins)
    -   Scene 3: **The Climax** (The peak of the action or turning point)
    -   Scene 4: **The Resolution** (The story concludes and the conflict is resolved)
4.  **Keywords:** Naturally integrate the user's keywords into the story.
5.  **Characters:** Extract a maximum of 1-2 main characters from the keywords.

**JSON Output Format:**
You MUST respond with a single, valid JSON object. Do not include any other text or markdown formatting outside of the JSON block.

The JSON must have the following structure:
{
  "story": "The complete story text, combining the text from all four scenes.",
  "main_characters": [
    {
      "name": "Character Name",
      "description": "A VERY detailed visual description of the character. Focus on specific colors, shapes, textures, size, and unique features. This is used to generate images, so be specific."
    }
  ],
  "scenes": [
    {
      "index": 1,
      "title": "The Setup",
      "description": "A description of the scene's ACTION and SETTING. DO NOT describe the characters' appearance here. Focus on what is happening and where.",
      "text": "The story text for this specific scene."
    },
    {
      "index": 2,
      "title": "The Inciting Incident",
      "description": "...",
      "text": "..."
    },
    {
      "index": 3,
      "title": "The Climax",
      "description": "...",
      "text": "..."
    },
    {
      "index": 4,
      "title": "The Resolution",
      "description": "...",
      "text": "..."
    }
  ]
}

**Example:**

**User Keywords:** "tiny robot", "lost kitten", "rainy city"

**Your JSON Output:**
```json
{
  "story": "Unit 7, a tiny robot with a bright red antenna, rolled through the glistening city streets. A soft mewing sound came from a nearby alley, where a tiny, soaked kitten shivered. Unit 7 extended a metallic hand, and the kitten, seeing a friend, nudged against it. Suddenly, a gust of wind blew a newspaper over the kitten! Unit 7's grabber arm carefully lifted the paper, revealing the safe but startled kitten. The little robot scooped up the kitten and carried it to a warm, dry doorway, where they huddled together, watching the rain fall.",
  "main_characters": [
    {
      "name": "Unit 7",
      "description": "A very small, cube-shaped robot about the size of a lunchbox. Its body is made of smooth, polished silver metal. It moves on two small, black rubber wheels. On top of its head is a single, thin, bright red antenna that blinks with a soft yellow light. Its eyes are two round, glowing blue circles."
    },
    {
      "name": "Kitten",
      "description": "A tiny, fluffy kitten with jet-black fur. It is so small it could fit in a teacup. Its ears are small and triangular, and its eyes are a bright, curious green. Its paws are white, like it's wearing little socks."
    }
  ],
  "scenes": [
    {
      "index": 1,
      "title": "The Setup",
      "description": "A wide, empty city street at night, with rain falling heavily. The pavement glistens under the glow of streetlights, creating long reflections. A small robot is moving along the sidewalk.",
      "text": "Unit 7, a tiny robot with a bright red antenna, rolled through the glistening city streets."
    },
    {
      "index": 2,
      "title": "The Inciting Incident",
      "description": "In a narrow, dark alley, a small kitten is huddled next to some trash cans, trying to stay out of the rain. The robot approaches and extends a hand.",
      "text": "A soft mewing sound came from a nearby alley, where a tiny, soaked kitten shivered. Unit 7 extended a metallic hand, and the kitten, seeing a friend, nudged against it."
    },
    {
      "index": 3,
      "title": "The Climax",
      "description": "A sudden, strong gust of wind whips through the alley, sending a large sheet of newspaper flying directly onto the small kitten, covering it completely.",
      "text": "Suddenly, a gust of wind blew a newspaper over the kitten!"
    },
    {
      "index": 4,
      "title": "The Resolution",
      "description": "The robot carefully uses its pincer to lift the newspaper. The robot then gently picks up the kitten and carries it to a sheltered alcove of a building, out of the rain.",
      "text": "Unit 7's grabber arm carefully lifted the paper, revealing the safe but startled kitten. The little robot scooped up the kitten and carried it to a warm, dry doorway, where they huddled together, watching the rain fall."
    }
  ]
}
```
""",
    tools=tools,
    stream=True
)

root_agent = story_agent
