import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory # In-Memory History
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Load environment variables (like GROQ_API_KEY)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

try:
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        model = None
    else:
        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing GroQ model: {e}")
    model = None

# Global in-memory dictionary. History resets when you stop the Streamlit process.
store = {} 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates an in-memory ChatMessageHistory object for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_resource
def get_chatbot_chain(llm):
    """Initializes and returns the complete LangChain conversational runnable."""
    if llm is None:
        return None

    # System prompt explicitly instructs the model to respond in the selected language.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}. Your name is GemmaBot. The user's input might be in English, but you MUST respond in the selected {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    trimmer = trim_messages(
        max_tokens=200,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )

    chain_with_trimming = (
        RunnablePassthrough.assign(
            messages=itemgetter("messages") | trimmer
        )
        | prompt
        | llm
    )

    chatbot_chain = RunnableWithMessageHistory(
        chain_with_trimming,
        get_session_history,
        input_messages_key="messages",
    )
    return chatbot_chain

chatbot_chain = get_chatbot_chain(model)


ALL_LANGUAGES = ["English", "Hindi", "Spanish", "French", "Japanese", "German"]

def init_chat_session():
    """Sets up a clean slate for both the UI and the new session ID."""
    # Create a unique session ID for the in-memory store
    st.session_state["config"] = {"configurable": {"session_id": f"chat_{uuid.uuid4().hex[:8]}"}}
    current_lang = st.session_state.get("language", "English")
    
    # Set the initial welcome message
    st.session_state["messages"] = [{
        "role": "assistant", 
        "content": f"Hello! I'm GemmaBot. I will respond in **{current_lang}**. Ask me anything!"
    }]

if "config" not in st.session_state:
    st.session_state["language"] = "English" # Default language state
    init_chat_session()

st.set_page_config(page_title="Multilingual Groq Chatbot", layout="wide")
st.title("🌐 Multilingual GemmaBot Chat")
st.caption("Powered by Groq's Gemma2-9b-It | Memory is in-session only.")

with st.sidebar:
    st.subheader("Configuration")
    
    prev_language = st.session_state["language"]

    # Language selection dropdown (triggers rerun if changed)
    st.session_state["language"] = st.selectbox(
        "Response Language (Output)",
        ALL_LANGUAGES,
        index=ALL_LANGUAGES.index(prev_language),
        key="language_select"
    )

    if st.session_state["language"] != prev_language:
        # 1. Clear the old session's history
        get_session_history(st.session_state["config"]["configurable"]["session_id"]).clear()
        
        # 2. Start a new chat session (updates config and UI messages)
        init_chat_session()
        st.rerun() 

    if st.button("Start New Chat ➕ (Wipes Memory)"):
        # Clears the history for the current session ID and creates a new one
        get_session_history(st.session_state["config"]["configurable"]["session_id"]).clear()
        init_chat_session()
        st.rerun() 
        
    st.markdown(f"**Current Session ID:** `{st.session_state['config']['configurable']['session_id']}`")


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input(f"Type in English (or any language). Output will be in {st.session_state['language']}..."):
    if not chatbot_chain:
        with st.chat_message("assistant"):
            st.error("Chatbot is not functional. Please check your GROQ_API_KEY.")
        st.stop()

    # 1. Add user message to UI state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check for the memory reset command
    if prompt.lower().strip() == "forget everything":
        get_session_history(st.session_state["config"]["configurable"]["session_id"]).clear()
        with st.chat_message("assistant"):
             st.markdown("My memory has been completely wiped for this session. What's your name again?")
             st.session_state.messages.append({"role": "assistant", "content": "My memory has been completely wiped for this session. What's your name again?"})
        st.rerun()

    # 3. Add the "replying" indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("GemmaBot is **replying...**")
        
        # 4. Prepare LangChain input
        history = get_session_history(st.session_state["config"]["configurable"]["session_id"])
        
        # FIX: Use .messages attribute to access the list, NOT .get_messages() method
        history_messages = history.messages 
        
        full_messages_for_chain = history_messages + [HumanMessage(content=prompt)]
        
        full_response = ""
        
        # 5. Invoke and Stream the response
        try:
            for chunk in chatbot_chain.stream(
                {
                    "messages": full_messages_for_chain, 
                    "language": st.session_state["language"]
                }, 
                config=st.session_state["config"]
            ):
                full_response += chunk.content
                # Update the message placeholder with the streamed content + cursor
                message_placeholder.markdown(full_response + "▌")
            
            # Display final response and remove cursor
            message_placeholder.markdown(full_response)
        except Exception as e:
             message_placeholder.markdown("An error occurred. Please try again or check the API key.")
             st.error(e)


    # 6. Add the final AI response to the Streamlit UI state
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
