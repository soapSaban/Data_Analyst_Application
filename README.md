# 📊 AI Data Analyst with Streamlit

**A powerful tool to analyze documents, extract insights, and visualize data using AI (Powered by Together.ai)**
## 🌟 Features

- **Multi-file support**: Upload PDFs, Word docs, Excel, CSV, images, and more
- **AI-powered analysis**: Automatic summarization and insights generation
- **Data visualization**: Auto-generated charts for numeric data
- **Q&A Chat**: Ask questions about your uploaded documents
- **Report generation**: Download PDF reports with key findings
- **Serverless AI models**: Choose from multiple LLM providers

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR (for image/PDF text extraction)
  ```bash
  # Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  # Mac: brew install tesseract
  # Linux: sudo apt install tesseract-ocr
- Together.ai API key (get it [here](https://together.ai))
- Create a `.env` file:
   ```env
   TOGETHER_API_KEY=your_key_here
### Installation
```bash

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

#Running
Import Tesseract-OCR/teserract.exe file path into the code
type < streamlit run app.py > in the terminal and smash Enter

🛠️ How It Works
Upload Files (PDF, DOCX, CSV, XLSX, TXT, JPG/PNG)

View AI-Generated:

Document summaries

Key insights

Data visualizations

Chat with your documents

Download reports as PDF

📂 File Support
FileType	TextExtraction	Data Analysis	OCR Support
PDF	              ✅	           ✅	      ✅ (beta)
Word(DOCX)	      ✅	           ✅	       -
Excel(XLSX)	      ✅	           ✅	       -
CSV	              ✅	           ✅	       -
Text(TXT)	        ✅	           ✅	       -
Images(JPG/PNG)   ✅	            -	      ✅

🤖 AI Models Supported
MistralAI/Mixtral-8x7B-Instruct-v0.1 (default, recommended)

MistralAI/Mistral-7B-Instruct-v0.1

TogetherComputer/LLaMA-2-7B-32K

Meta-Llama/Llama-3-8b-chat-hf

Change models in the app sidebar

📜 License
MIT License - See LICENSE for details

🙏 Acknowledgments
Powered by Together.ai

Built with Streamlit

Uses LangChain for document processing

