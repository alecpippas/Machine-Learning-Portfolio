# Autoregressive n-Gram Text Generator

## Project Overview

This project implements a sophisticated text generation system using autoregressive n-gram models, demonstrating the fundamental principles behind modern language models like GPT. The system analyzes literary texts from Project Gutenberg to learn probabilistic patterns and generate coherent text that mimics the style and structure of the source material.

**What it does:** The application downloads and preprocesses classic literature, builds probabilistic n-gram models of varying complexity, generates text with configurable parameters, and evaluates the quality vs. creativity trade-off using sophisticated statistical analysis.

**Why it's interesting:** This project showcases the mathematical foundations of modern language generation, demonstrating how probabilistic modeling can capture linguistic patterns. It provides insights into the fundamental trade-offs between model complexity, text quality, and creative originality that are crucial for understanding current AI language models.

## Key Features Implemented

### Core Functionality
- **Text Preprocessing**: Automated download and cleaning of Project Gutenberg texts
- **N-Gram Modeling**: Implementation of Markov Chain-based text generation
- **Multi-Order Analysis**: Support for various n-gram orders (1-gram through 6-gram)
- **Text Generation**: Configurable paragraph generation with seed text selection
- **Quality Evaluation**: Longest Common Substring (LCS) analysis for plagiarism detection

### Advanced Features
- **Statistical Analysis**: Comprehensive evaluation of model performance across multiple texts
- **Visualization**: Professional plotting of LCS statistics with quartile analysis
- **Multi-Book Support**: Analysis of Shakespeare, Kafka, Carroll, and Dumas works
- **Performance Optimization**: Efficient frequency dictionary implementation
- **Creative Enhancement**: Bonus model improvements for better text quality

### Technical Implementation
- **Data Pipeline**: Automated text download and preprocessing workflow
- **Memory Management**: Efficient handling of large literary texts
- **Statistical Computing**: NumPy-based statistical analysis and visualization
- **Error Handling**: Robust text processing with metadata removal

## Technologies Used

### Core Libraries
- **NumPy**: Numerical computing and statistical analysis
- **Matplotlib**: Professional data visualization and plotting
- **tqdm**: Progress tracking and user experience enhancement
- **re (Regex)**: Advanced text pattern matching and cleaning

### Data Sources
- **Project Gutenberg**: Free public domain literary texts
- **Classic Literature**: Shakespeare, Kafka, Carroll, Dumas works
- **Text Processing**: UTF-8 encoding support for international texts

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis
- **Python 3.x**: Modern Python with advanced language features
- **Statistical Analysis**: Built-in statistical computing capabilities

## How to Run the Project

### Prerequisites
- Python 3.7 or higher
- Internet connection for Project Gutenberg downloads
- Sufficient disk space for literary texts (approximately 50-100 MB)

### Installation
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `Books/` directory exists for downloaded texts

### Running the Application

#### Basic Text Generation
```python
# Load and preprocess a book
shakespeare = word_sequence_from_file("Books/Shakespeare.txt")

# Create a 3-gram model
freq_dict = make_freq_dict(3, shakespeare)

# Generate text
start_text = ["To", "be", "or"]
generated_text = predict_paragraph(start_text, 3, freq_dict, gen_length=100)
print(" ".join(generated_text))
```

#### Advanced Analysis
```python
# Generate LCS statistics for model evaluation
lcs_stats = gen_lcs_statistics(shakespeare)

# Visualize performance across different n-gram orders
plot_lcs_statistics(shakespeare, "Shakespeare Complete Works", lcs_stats)

# Compare multiple books
plot_many_lcs_statistics([
    {"book_name": "Shakespeare", "stats": shakes_stats},
    {"book_name": "Kafka", "stats": kafka_stats}
])
```

### Usage Examples

#### Single Book Analysis
```python
# Download and analyze Alice in Wonderland
wonderland = word_sequence_from_file("Books/Wonderland.txt")
freq_dict_2 = make_freq_dict(2, wonderland)

# Generate creative text
start_phrase = ["Alice", "was"]
generated_story = predict_paragraph(start_phrase, 2, freq_dict_2, gen_length=200)
```

#### Multi-Book Comparison
```python
# Analyze text quality vs. creativity trade-offs
books = [shakespeare, metamorphosis, wonderland, montecristo]
book_names = ["Shakespeare", "Kafka", "Carroll", "Dumas"]

for book, name in zip(books, book_names):
    stats = gen_lcs_statistics(book)
    plot_lcs_statistics(book, name, stats)
```

## Project Structure

```
Autoregressive_n-Gram_Text_Generator/
├── autoregressive_markov_n-gram_text_enerator.ipynb  # Main implementation
├── requirements.txt                                   # Python dependencies
├── README.md                                         # Project documentation
└── training_text/                                            # Downloaded literary texts
    ├── Shakespeare.txt                               # Complete Shakespeare works
    ├── Metamorphosis.txt                             # Kafka's Metamorphosis
    ├── Wonderland.txt                                # Alice in Wonderland
    └── MonteCristo.txt                              # The Count of Monte Cristo
```

## Technical Implementation Details

### N-Gram Model Architecture
- **Markov Chain Implementation**: Order k-1 Markov Chain for k-gram models
- **Frequency Dictionary**: Efficient storage of conditional probabilities
- **Text Generation**: Probabilistic word selection based on context

### Text Preprocessing Pipeline
1. **Download**: Automated retrieval from Project Gutenberg
2. **Cleaning**: Metadata and legal text removal
3. **Tokenization**: Word-level text splitting and normalization
4. **Validation**: Text integrity verification and error handling

### Statistical Analysis Framework
- **LCS Calculation**: Longest Common Substring analysis for quality assessment
- **Quartile Analysis**: Statistical distribution analysis across model orders
- **Performance Metrics**: Quality vs. creativity trade-off quantification

## Key Insights and Learnings

### Mathematical Concepts
- **Markov Processes**: Understanding state-dependent probability systems
- **Conditional Probability**: Modeling word dependencies in natural language
- **Statistical Distributions**: Analysis of text generation quality metrics

### NLP Fundamentals
- **Language Modeling**: Foundation of modern text generation systems
- **Text Preprocessing**: Real-world challenges in literary text handling
- **Quality Evaluation**: Metrics for assessing generated text originality

### Machine Learning Principles
- **Model Complexity Trade-offs**: Balancing accuracy with generalization
- **Data Quality Impact**: How preprocessing affects model performance
- **Evaluation Methodologies**: Statistical approaches to model assessment

## Performance Analysis

### Model Order Impact
- **Low Order (1-2)**: High creativity, low coherence
- **Medium Order (3-4)**: Balanced creativity and coherence
- **High Order (5-6)**: High coherence, potential plagiarism

### Statistical Findings
- **Exponential Growth**: LCS length grows exponentially with model order
- **Quality Plateau**: Diminishing returns beyond 4-gram models
- **Book Variations**: Different literary styles show varying performance patterns

### Optimization Strategies
- **Memory Efficiency**: Optimized frequency dictionary implementation
- **Computational Complexity**: Efficient n-gram generation algorithms
- **Scalability**: Support for large literary texts and multiple books

## Practical Applications

This foundational knowledge applies to:
- **Language Model Development**: Understanding modern AI text generation
- **NLP Research**: Foundation for advanced language processing
- **Content Generation**: Automated text creation for various domains
- **Educational Tools**: Language learning and creative writing assistance
- **AI Ethics**: Understanding plagiarism and originality in generated content

## Future Enhancements

Potential improvements for this project:
- **Neural Network Integration**: Modern transformer-based approaches
- **Advanced Sampling**: Temperature and top-k sampling techniques
- **Multi-Language Support**: Extension to non-English texts
- **Real-time Generation**: Interactive text generation interfaces
- **Style Transfer**: Cross-author style imitation capabilities
- **Evaluation Metrics**: BLEU, ROUGE, and other NLP metrics
- **Web Interface**: User-friendly web application for text generation

## Troubleshooting

### Common Issues
1. **Download Failures**: Check internet connection and Project Gutenberg availability
2. **Memory Errors**: Large texts may require sufficient RAM
3. **Text Encoding**: Ensure UTF-8 support for international characters
4. **File Paths**: Verify Books/ directory exists and is accessible

### Debug Tips
- Monitor download progress with tqdm indicators
- Check text preprocessing results with sample outputs
- Verify frequency dictionary construction with small test cases
- Use smaller text samples for initial testing

## Contributing

This project demonstrates practical implementation of autoregressive text generation using classical probabilistic methods. Contributions and improvements are welcome, particularly in:
- Advanced sampling techniques
- Additional evaluation metrics
- Performance optimization
- Multi-language support
- Web interface development

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Project Gutenberg**: For providing free access to public domain literature
- **Classical Authors**: Shakespeare, Kafka, Carroll, and Dumas for literary works
- **Academic Community**: For advancing probabilistic language modeling research
- **Open Source Community**: For the libraries and tools that enable this research
