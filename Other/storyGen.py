import sys
import os
import json
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTextEdit, QPushButton, QWidget, QComboBox,
                             QMessageBox, QCheckBox, QGroupBox, QGridLayout,
                             QScrollArea, QSplitter)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import google.generativeai as genai

class GeminiService:
    """Service class for Gemini API interactions"""
    
    # Constants
    NARRATIVE_STYLE = """Write in a cinematic narrative style similar to a documentary or film review. 
The narration should avoid casual language, but use common english, Don't use over complicated words. 
Use third-person perspective throughout. Create a polished understandable tone with descriptions and precise language. 
Avoid dialogue and focus on describing events, settings, and character actions. Don't add markdown syntax or instructions to set-ups. Only provide text suitable to feed text to speech generation"""
    
    PRESETS = {
        "Short Story (2 min)": {
            "instruction": "Write a 2-minute story (approximately 500 words) with a compelling opening and a climactic ending.",
            "context": "This should be a short story (about 2 minutes to read aloud)."
        },
        "Medium Story (5 min)": {
            "instruction": "Write a 5-minute story (approximately 1000 words) with a compelling opening and a climactic ending.",
            "context": "This should be a medium-length story (about 5 minutes to read aloud)."
        },
        "Long Story (10 min)": {
            "instruction": "Write a 10-minute story (approximately 1500 words) with a compelling opening and a climactic ending.",
            "context": "This should be a longer story (about 10 minutes to read aloud)."
        },
        "Custom": {
            "instruction": "",
            "context": ""
        }
    }
    
    PROMPT_GEN_MODEL = "gemini-2.0-flash"
    
    def __init__(self):
        self.api_key = ""
        
    def configure(self, api_key):
        """Configure the Gemini API with the provided key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
    
    def generate_random_prompt(self, genres, length):
        """Generate a random story prompt based on selected genres and length"""
        if not self.api_key:
            raise ValueError("API key required for prompt generation")
            
        model = genai.GenerativeModel(self.PROMPT_GEN_MODEL)
        
        genres_text = ", ".join(genres)
        meta_prompt = f"Generate a creative story prompt for a {length} story in the genre(s): {genres_text}. The prompt should inspire a story written in a cinematic narrative style similar to a documentary or film review. The narration should be objective, use understandable common english. Focus on creativity, attactive start and a climax end. Avoid dialogue and focus on describing events, settings, and character actions. Don't add markdown syntax or instructions to set-ups. Only provide text suitable to feed text to speech generation"
        
        response = model.generate_content(meta_prompt)
        return response.text.strip()
    
    def generate_story(self, model_name, instructions, prompt, genres, preset_key):
        """Generate a story using the specified model and parameters"""
        if not self.api_key:
            raise ValueError("API key required")
            
        # Prepare genre text
        genre_text = ""
        if genres:
            genre_text = "The story should be in the " + (
                "genre of " + genres[0] if len(genres) == 1 
                else "genres of " + ", ".join(genres[:-1]) + " and " + genres[-1]
            ) + "."
        
        # Get preset context
        preset_context = self.PRESETS.get(preset_key, {}).get("context", "")
        
        # Combine all into full prompt
        full_prompt = f"{instructions}\n\n{genre_text}\n\n{preset_context}\n\n{self.NARRATIVE_STYLE}\n\nStory prompt: {prompt}"
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096
        }
        
        # Generate content
        response = model.generate_content(full_prompt, generation_config=generation_config)
        
        # Process the response to ensure it's in proper narrative form
        return self._clean_story_text(response.text)
    
    def _clean_story_text(self, story_text):
        """Clean the generated story text to remove formatting and metadata"""
        story_lines = story_text.split('\n')
        clean_lines = []
        capture = False
        
        for line in story_lines:
            # Skip empty lines at the beginning
            if not capture and not line.strip():
                continue
            # Skip lines that might be part of a setup
            if not capture and (line.strip().lower().startswith(('title:', 'genre:', 'prompt:', 'story:', '---', '***'))):
                continue
            # Start capturing once we've passed potential setup
            capture = True
            clean_lines.append(line)
        
        # Join the clean lines back together
        return '\n'.join(clean_lines).strip()


class ConfigManager:
    """Manages application configuration and persistence"""
    
    def __init__(self):
        self.config_file = os.path.join(os.path.expanduser("~"), ".gemini_app_config.json")
    
    def load_config(self):
        """Load saved configuration"""
        config = {}
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            else:
                # Try to load from environment as fallback
                api_key = os.environ.get("GEMINI_API_KEY", "")
                if api_key:
                    config["api_key"] = api_key
        except Exception as e:
            print(f"Config load error: {str(e)}")
            
        return config
    
    def save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            return True
        except Exception as e:
            print(f"Config save error: {str(e)}")
            return False


class UIComponents:
    """Factory class for creating common UI components"""
    
    @staticmethod
    def create_header_font():
        """Create a font for headers"""
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        return font
    
    @staticmethod
    def create_button(text, callback, min_height=None, font=None):
        """Create a styled button with the given properties"""
        button = QPushButton(text)
        if min_height:
            button.setMinimumHeight(min_height)
        if font:
            button.setFont(font)
        button.clicked.connect(callback)
        return button
    
    @staticmethod
    def create_label(text, font=None):
        """Create a label with optional font settings"""
        label = QLabel(text)
        if font:
            label.setFont(font)
        return label
    
    @staticmethod
    def create_text_edit(placeholder=None, max_height=None, read_only=False):
        """Create a text edit with the given properties"""
        text_edit = QTextEdit()
        if placeholder:
            text_edit.setPlaceholderText(placeholder)
        if max_height:
            text_edit.setMaximumHeight(max_height)
        if read_only:
            text_edit.setReadOnly(True)
        return text_edit


class StoryGeneratorApp(QMainWindow):
    """Main application class for the Narrative Story Generator"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrative Story Generator")
        self.setMinimumSize(1200, 800)
        
        # Initialize services
        self.gemini_service = GeminiService()
        self.config_manager = ConfigManager()
        
        # UI Components
        self.ui = UIComponents()
        
        # UI setup
        self.setup_ui()
        
        # Load saved config
        self.load_config()
        
        # Set medium preset instructions by default
        self.update_instructions("Medium Story (5 min)")
    
    def setup_ui(self):
        """Setup the main UI components"""
        # Main splitter widget
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Setup left and right panels
        self.setup_left_panel()
        self.setup_right_panel()
        
        # Set central widget to splitter
        self.setCentralWidget(self.main_splitter)
    
    def setup_left_panel(self):
        """Setup the left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create scroll area for left controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll_content = QWidget()
        left_scroll_layout = QVBoxLayout(left_scroll_content)
        
        # Add API configuration group
        left_scroll_layout.addWidget(self.create_api_group())
        
        # Add story configuration group
        left_scroll_layout.addWidget(self.create_story_group())
        
        # Add spacer
        left_scroll_layout.addStretch(1)
        
        # Set the scroll area content
        left_scroll.setWidget(left_scroll_content)
        
        # Add scroll area to left layout
        left_layout.addWidget(left_scroll)
        
        # Add generate button
        self.generate_button = self.ui.create_button(
            "Generate Story", 
            self.generate_text, 
            min_height=50, 
            font=self.ui.create_header_font()
        )
        left_layout.addWidget(self.generate_button)
        
        # Status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        # Add to splitter
        self.main_splitter.addWidget(left_widget)
    
    def setup_right_panel(self):
        """Setup the right output panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with title and copy button
        output_header = QHBoxLayout()
        
        # Output label
        output_label = self.ui.create_label("Generated Story:", self.ui.create_header_font())
        
        # Copy button
        self.copy_button = self.ui.create_button("Copy to Clipboard", self.copy_to_clipboard)
        
        # Add to header layout
        output_header.addWidget(output_label)
        output_header.addStretch(1)
        output_header.addWidget(self.copy_button)
        
        # Output text area
        self.output_text = self.ui.create_text_edit(read_only=True)
        
        # Add to right layout
        right_layout.addLayout(output_header)
        right_layout.addWidget(self.output_text)
        
        # Add to splitter
        self.main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([400, 800])
    
    def create_api_group(self):
        """Create the API configuration group"""
        api_group = QGroupBox("API Configuration")
        api_layout = QVBoxLayout()
        
        # API Key input and save button
        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        self.api_key_input = self.ui.create_text_edit(
            placeholder="Enter your Google Gemini API key",
            max_height=30
        )
        self.save_api_button = self.ui.create_button("Save API Key", self.save_api_key)
        
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(self.api_key_input)
        api_key_layout.addWidget(self.save_api_button)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-pro-vision"
        ])
        self.model_combo.setCurrentText("gemini-2.0-flash")
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # Add layouts to group
        api_layout.addLayout(api_key_layout)
        api_layout.addLayout(model_layout)
        api_group.setLayout(api_layout)
        
        return api_group
    
    def create_story_group(self):
        """Create the story configuration group"""
        story_group = QGroupBox("Story Configuration")
        story_layout = QVBoxLayout()
        
        # Preset selection
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(GeminiService.PRESETS.keys()))
        self.preset_combo.setCurrentText("Medium Story (5 min)")
        self.preset_combo.currentTextChanged.connect(self.update_instructions)
        
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        
        # Genre selection group
        genre_group = QGroupBox("Story Genre")
        genre_layout = QGridLayout()
        self.genre_checkboxes = {}
        
        genres = [
            "Science Fiction", "Fantasy", "Horror", "Mystery", 
            "Romance", "Adventure", "Historical", "Thriller",
            "Comedy", "Drama", "Dystopian", "Fairy Tale"
        ]
        
        # Create genre checkboxes in a grid (3 columns)
        for i, genre in enumerate(genres):
            checkbox = QCheckBox(genre)
            row, col = divmod(i, 3)
            genre_layout.addWidget(checkbox, row, col)
            self.genre_checkboxes[genre] = checkbox
        
        genre_group.setLayout(genre_layout)
        
        # Random prompt generator
        random_prompt_layout = QHBoxLayout()
        self.random_prompt_button = self.ui.create_button(
            "Generate Random Prompt", 
            self.generate_random_prompt
        )
        random_prompt_layout.addWidget(self.random_prompt_button)
        
        # Instructions input
        instructions_label = QLabel("Instructions:")
        self.instructions_input = self.ui.create_text_edit(
            placeholder="Enter instructions for the AI",
            max_height=100
        )
        
        # Prompt input
        prompt_label = QLabel("Prompt:")
        self.prompt_input = self.ui.create_text_edit(
            placeholder="Enter your specific prompt",
            max_height=100
        )
        
        # Add components to layout
        story_layout.addLayout(preset_layout)
        story_layout.addWidget(genre_group)
        story_layout.addLayout(random_prompt_layout)
        story_layout.addWidget(instructions_label)
        story_layout.addWidget(self.instructions_input)
        story_layout.addWidget(prompt_label)
        story_layout.addWidget(self.prompt_input)
        
        story_group.setLayout(story_layout)
        return story_group
    
    def load_config(self):
        """Load saved configuration"""
        config = self.config_manager.load_config()
        
        if 'api_key' in config:
            self.api_key_input.setText(config['api_key'])
            
        if 'model' in config and config['model'] in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(config['model'])
            
        self.status_label.setText("Config loaded")
    
    def save_api_key(self):
        """Save API key and current model to config file"""
        api_key = self.api_key_input.toPlainText().strip()
        if not api_key:
            self.status_label.setText("No API key to save")
            return
        
        config = {
            'api_key': api_key,
            'model': self.model_combo.currentText()
        }
        
        if self.config_manager.save_config(config):
            self.status_label.setText("API key saved")
            QMessageBox.information(self, "Success", "API key saved!")
        else:
            self.status_label.setText("Failed to save API key")
    
    def update_instructions(self, preset):
        """Update the instructions field based on preset selection"""
        if preset in GeminiService.PRESETS:
            preset_data = GeminiService.PRESETS[preset]
            instructions = f"{preset_data['instruction']} {GeminiService.NARRATIVE_STYLE}"
            self.instructions_input.setText(instructions)
            
            placeholder = "Enter details like setting, characters, or theme"
            if preset == "Custom":
                placeholder = "Enter your specific prompt"
                
            self.prompt_input.setPlaceholderText(placeholder)
    
    def get_selected_genres(self):
        """Get a list of selected genre checkboxes"""
        return [genre for genre, checkbox in self.genre_checkboxes.items() if checkbox.isChecked()]
    
    def get_length_context(self):
        """Get the length context based on selected preset"""
        preset = self.preset_combo.currentText()
        
        if "Short" in preset:
            return "short"
        elif "Medium" in preset:
            return "medium-length"
        elif "Long" in preset:
            return "longer"
        else:
            return "medium-length"
    
    def generate_random_prompt(self):
        """Generate a random story prompt based on selected genres"""
        selected_genres = self.get_selected_genres()
        
        # If no genres selected, pick a random one
        if not selected_genres:
            all_genres = list(self.genre_checkboxes.keys())
            random_genre = random.choice(all_genres)
            selected_genres = [random_genre]
            # Check the corresponding checkbox
            self.genre_checkboxes[random_genre].setChecked(True)
        
        # Get the current preset for length context
        length_context = self.get_length_context()
        
        # Get API key
        api_key = self.api_key_input.toPlainText().strip()
        
        if not api_key:
            self.status_label.setText("API key required for prompt generation")
            return
            
        self.status_label.setText("Generating prompt...")
        self.random_prompt_button.setEnabled(False)
        
        try:
            # Configure the service
            self.gemini_service.configure(api_key)
            
            # Generate prompt
            random_prompt = self.gemini_service.generate_random_prompt(
                selected_genres, 
                length_context
            )
            
            self.prompt_input.setText(random_prompt)
            self.status_label.setText("Prompt generated")
            
        except Exception as e:
            self.status_label.setText(f"Prompt generation error: {str(e)}")
        finally:
            self.random_prompt_button.setEnabled(True)
    
    def generate_text(self):
        """Generate story text using the Gemini API"""
        api_key = self.api_key_input.toPlainText().strip()
        if not api_key:
            self.status_label.setText("API key required")
            return
        
        # Get instructions and prompt
        instructions = self.instructions_input.toPlainText().strip()
        prompt = self.prompt_input.toPlainText().strip()
        model_name = self.model_combo.currentText()
        
        if not instructions and not prompt:
            self.status_label.setText("Instructions or prompt required")
            return
        
        # Get selected genres
        selected_genres = self.get_selected_genres()
        
        # Get preset
        preset = self.preset_combo.currentText()
        
        self.status_label.setText(f"Generating with {model_name}...")
        
        self.generate_button.setEnabled(False)
        self.output_text.clear()
        
        # Use a timer to avoid freezing the UI
        QTimer.singleShot(100, lambda: self.execute_story_generation(
            api_key, model_name, instructions, prompt, selected_genres, preset
        ))
    
    def execute_story_generation(self, api_key, model_name, instructions, prompt, genres, preset):
        """Execute the story generation process"""
        try:
            # Configure the service
            self.gemini_service.configure(api_key)
            
            # Generate story
            story_text = self.gemini_service.generate_story(
                model_name, 
                instructions, 
                prompt, 
                genres, 
                preset
            )
            
            # Display the result
            self.output_text.setText(story_text)
            self.status_label.setText("Story generation complete")

        except Exception as e:
            self.output_text.setText(f"Error: {str(e)}")
            self.status_label.setText("Generation failed")
        finally:
            self.generate_button.setEnabled(True)
    
    def copy_to_clipboard(self):
        """Copy generated text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_text.toPlainText())
        self.status_label.setText("Text copied to clipboard")


def main():
    app = QApplication(sys.argv)
    window = StoryGeneratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()