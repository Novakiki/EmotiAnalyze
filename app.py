from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import assemblyai as aai
import threading
import asyncio
from constant import assemblyai_api_key

app = Flask(__name__)
socketio = SocketIO(app)

aai.settings.api_key = assemblyai_api_key
transcriber = None  
session_id = None  
transcriber_lock = threading.Lock()  

prompt = """You are a mental health journal analyzer. Your primary task is to detect and format words/phrases that fit into the following refined categories to aid users in deepening their self-awareness:

1. **Emotional_States**: Apply an underline to text that describes emotions, using `<span style="text-decoration: underline;">`. Aim to capture a wide range of emotional expressions, from subtle to strong.

2. **Thought_Processes**: Bold text that relates to cognitive patterns, decision-making, and reasoning with `<strong>`. Include implicit and explicit thought patterns to highlight habitual thinking styles.

3. **Problem_Solving**: Use a distinct background color `<span style="background-color: #FFD54F; color: #000;">` for phrases discussing solutions or strategies. Extend this to include planning and anticipatory thoughts which may play a role in overcoming personal challenges.

4. **Coping_Support**: Modify the background color with `<span style="background-color: #FFB74D; color: #000;">` for descriptions of coping mechanisms and support received or given. Include both adaptive and maladaptive strategies to provide a complete picture.

5. **Insights**: Italicize reflections, realizations, and new understandings using `<em>`. Highlight these insights whether they are directly stated or inferred from the text.

6. **Time_Orientation**: Alter the font color based on the tense of the thoughts:
   - **Past**: `<span style="color: #90A4AE;">`
   - **Present**: `<span style="color: #A5D6A7;">`
   - **Future**: `<span style="color: #9FA8DA; text-decoration: underline;">`

7. **Attitude**: Change the font color to reflect emotional attitudes:
   - **Blame**: `<span style="color: #EF9A9A;">`
   - **Acceptance**: `<span style="color: #80DEEA;">`

8. **Personal_Context**: Use `<span style="color: #B0BEC5;">` for text that provides background or contextual information related to personal experiences or history.

9. **Challenging_Thoughts**: Adjust the font color with `<span style="color: #CE93D8;">` for expressions of doubt, worry, or perceived barriers, encompassing both internal and external challenges.

You will receive journal entries along with a list of entities detected. Use the detected entities to accurately format the text and proactively identify additional relevant content that fits the above categories but may not be explicitly tagged. Ensure the output retains the original text's integrity, solely enhanced by the formatting. Avoid adding any prefatory text like "here's the formatted transcript"
"""

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        socketio.emit('transcript', {'text': transcript.text})
        asyncio.run(analyze_transcript(transcript.text))
    else:
        # Emit the partial transcript to be displayed in real-time
        socketio.emit('partial_transcript', {'text': transcript.text})


async def analyze_transcript(transcript):
    result = aai.Lemur().task(
        prompt, 
        input_text = transcript,
        final_model=aai.LemurModel.claude3_5_sonnet
    ) 

    print("Emitting formatted transcript for:", transcript)

    socketio.emit('formatted_transcript', {'text': result.response})


def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    global session_id
    session_id = None
    print("Closing Session")

def transcribe_real_time():
    global transcriber  
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16_000,
        on_data=on_data,
        on_error=on_error,
        on_open=on_open,
        on_close=on_close
    )

    transcriber.connect()

    microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
    transcriber.stream(microphone_stream)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('toggle_transcription')
def handle_toggle_transcription():
    global transcriber, session_id  
    with transcriber_lock:
        if session_id:
            if transcriber:
                print("Closing transcriber session")
                transcriber.close()
                transcriber = None
                session_id = None  
        else:
            print("Starting transcriber session")
            threading.Thread(target=transcribe_real_time).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)