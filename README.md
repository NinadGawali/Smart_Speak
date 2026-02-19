# Smart_Speak
Smart Speak is an innovative application designed to analyze and evaluate speech recordings. It provides users with insights into their speaking style, including aspects such as tone, pace, clarity, and emotional expression. By leveraging advanced speech analysis techniques, Smart Speak helps users improve their communication skills and gain confidence in their speaking abilities. Whether you're preparing for a presentation, practicing public speaking, or simply want to enhance your verbal communication, Smart Speak offers valuable feedback to help you become a more effective speaker.


## Instructions to Run the App
1. **Clone the Repository**: Start by cloning the Smart Speak repository to your local machine.

   ```bash
   git clone https://github.com/NinadGawali/Smart_Speak
   ```
2. **Navigate to the Project Directory**: Change your current directory to the Smart Speak project folder.

   ```bash
    cd Smart_Speak
    ```
4. **Set Up Environment Variables**: Create a `.env` file in the root directory of the project and add your API keys or any necessary environment variables. For example:
    ```
    GEMINI_API_KEY = "your_api_key_here"
    ```
5. **Create a Virtual Environment**: It is recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   venv/bin/activate  
   ```

3. **Install Dependencies**: Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```
6. **Run the Application**: Start the Streamlit application.
    ```bash
    streamlit run llm_inference/speech_info_analysis.py
    ```
7. **Access the App**: Open your web browser and navigate to `http://localhost:8501` to access the Smart Speak application.

## Notes
- Ensure you have the necessary API keys and permissions to use any external services integrated into the application.
- The application needs some time to load at first, especially if it is downloading models or dependencies. Please be patient while it initializes.
- First enter your content that you are delivering then record your speech and then click on the "Transcribe & Analyse" button to get insights into your speaking style.