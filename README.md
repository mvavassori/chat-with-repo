# Chat with Repo

## Description

A Streamlit application for creating a chatbot that can interact with codebases.

The application allows users to select an OpenAI GPT model, enter a GitHub repository URL, and specify an Activeloop dataset URL. It then loads the codebase from the GitHub repository, splits the code into chunks, creates a vector store, and initializes a conversation chain using the selected GPT model. Users can then start chatting with the chatbot.

## Installation

To install and set up the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/mvavassori/chat-with-repo
```

2. Navigate to the project directory:

```
cd <project_directory>
```

3. (Optional but recommended) Create a virtual environment:

```
python -m venv myenv
```

4. Activate the virtual environment:

- Windows:
  ```
  myenv\Scripts\activate
  ```
- macOS and Linux:
  ```
  source myenv/bin/activate
  ```

5. Install the project dependencies:

```
pip install -r requirements.txt
```

6. Make a `.env` file and paste inside it your [Activeloop](https://app.activeloop.ai/) token and [OpenAI](https://platform.openai.com/account/api-keys) API key

```
OPENAI_API_KEY=add_here_your_api_key
ACTIVELOOP_TOKEN=add_here_your_api_token
```

## Usage

To run the app in the browser using [Streamlit](https://streamlit.io/) (a simple web app builder for python) go to the root directory of the application and run:

```
streamlit run app.py
```

and click on the localhost link in the terminal.

Once you'll see the web app open in the browser follow the instructions to the left side of the page to create a new vector store from a GitHub repo and load it to Activeloop, or just to load an existing Activeloop vector store.
