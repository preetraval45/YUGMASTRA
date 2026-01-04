"""
Custom NLP Engine for Security Report Generation and Analysis
Bidirectional LSTM with Attention for threat analysis and report generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class CustomTokenizer:
    """
    Custom tokenizer for cybersecurity text
    Handles security-specific terminology
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_built = False

        # Security-specific tokens
        self.security_tokens = [
            "CVE", "MITRE", "ATT&CK", "CVSS", "exploit", "vulnerability",
            "malware", "ransomware", "phishing", "SQL injection", "XSS",
            "buffer overflow", "privilege escalation", "lateral movement",
            "command and control", "C2", "IOC", "indicator", "threat actor",
            "APT", "zero-day", "payload", "shellcode", "backdoor", "trojan"
        ]

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts"""
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Add high-frequency tokens to vocab
        idx = len(self.word2idx)
        for token, count in token_counts.most_common(self.vocab_size):
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        self.vocab_built = True
        logger.info(f"✅ Vocabulary built: {len(self.word2idx)} tokens")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Lowercase and clean
        text = text.lower()

        # Handle special security terms
        for term in self.security_tokens:
            text = text.replace(term.lower(), term.lower().replace(" ", "_"))

        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def encode(self, text: str, max_len: Optional[int] = None, add_special: bool = True) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)

        if add_special:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]

        # Convert to IDs
        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

        # Pad or truncate
        if max_len is not None:
            if len(ids) < max_len:
                ids = ids + [self.word2idx["<PAD>"]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text"""
        tokens = [self.idx2word.get(idx, "<UNK>") for idx in ids]

        if skip_special:
            tokens = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]]

        # Rejoin security terms
        text = " ".join(tokens)
        for term in self.security_tokens:
            text = text.replace(term.lower().replace(" ", "_"), term)

        return text


class AttentionLayer(nn.Module):
    """
    Attention mechanism for focusing on relevant parts of input
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch_size, seq_len, hidden_size * 2)
            hidden: (batch_size, hidden_size * 2)

        Returns:
            context: (batch_size, hidden_size * 2)
            attention_weights: (batch_size, seq_len)
        """
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state for each time step
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention scores
        energy = torch.tanh(self.attention(encoder_outputs + hidden_repeated))
        attention_scores = self.context_vector(energy).squeeze(2)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)

        return context, attention_weights


class BidirectionalLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for processing security text
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len)

        Returns:
            outputs: (batch_size, seq_len, hidden_size * 2)
            (hidden, cell): Final hidden and cell states
        """
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Attention for generating security reports
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.attention = AttentionLayer(hidden_size)

        # LSTM input: embedding + context
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size * 2,
            hidden_size * 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (h, c) each (num_layers, batch_size, hidden_size * 2)
            encoder_outputs: (batch_size, seq_len, hidden_size * 2)

        Returns:
            output: (batch_size, vocab_size)
            hidden: Updated hidden state
            attention_weights: (batch_size, seq_len)
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))  # (batch_size, 1, embedding_dim)

        # Calculate attention
        context, attention_weights = self.attention(encoder_outputs, hidden[0][-1])

        # Concatenate embedding and context
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)

        # Pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)

        # Generate prediction
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Sequence-to-Sequence model with Attention
    For threat analysis and security report generation
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()

        self.encoder = BidirectionalLSTMEncoder(
            vocab_size, embedding_dim, hidden_size, num_layers, dropout, padding_idx
        )
        self.decoder = AttentionDecoder(
            vocab_size, embedding_dim, hidden_size, num_layers, dropout, padding_idx
        )

        self.padding_idx = padding_idx
        logger.info(f"✅ Seq2Seq with Attention initialized: {embedding_dim}d embeddings, {hidden_size}d hidden")

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: (batch_size, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # Initialize decoder hidden state
        decoder_hidden = (hidden, cell)

        # Initialize output tensor
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        # First input to decoder is <SOS> token
        input_token = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            # Decode
            output, decoder_hidden, _ = self.decoder(input_token, decoder_hidden, encoder_outputs)
            outputs[:, t] = output

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = output.argmax(1).unsqueeze(1)

        return outputs

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        sos_token: int = 2,
        eos_token: int = 3,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate security report from input

        Args:
            src: (batch_size, src_len) - threat indicators
            max_len: Maximum length of generated report
            sos_token: Start of sequence token ID
            eos_token: End of sequence token ID
            temperature: Sampling temperature

        Returns:
            generated: (batch_size, generated_len)
            attention_weights: List of attention weight tensors
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_hidden = (hidden, cell)

        # Initialize with <SOS> token
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        attention_weights_list = []

        for _ in range(max_len):
            # Decode
            output, decoder_hidden, attention_weights = self.decoder(
                generated[:, -1].unsqueeze(1),
                decoder_hidden,
                encoder_outputs
            )

            # Apply temperature
            output = output / temperature

            # Sample next token
            probs = F.softmax(output, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            attention_weights_list.append(attention_weights)

            # Stop if all sequences generated <EOS>
            if (next_token == eos_token).all():
                break

        return generated, attention_weights_list


class CustomNLPEngine:
    """
    High-level NLP engine for security text processing
    """

    def __init__(self, vocab_size: int = 10000):
        self.tokenizer = CustomTokenizer(vocab_size)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(
        self,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """Initialize the seq2seq model"""
        self.model = Seq2SeqWithAttention(
            vocab_size=len(self.tokenizer.word2idx),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=self.tokenizer.word2idx["<PAD>"]
        ).to(self.device)

        logger.info(f"✅ NLP Engine initialized on {self.device}")

    def train_vocabulary(self, texts: List[str]):
        """Build vocabulary from training texts"""
        self.tokenizer.build_vocab(texts)

    def generate_security_report(
        self,
        threat_indicators: str,
        max_len: int = 200,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate security report from threat indicators

        Args:
            threat_indicators: Input text describing threats
            max_len: Maximum report length
            temperature: Generation temperature

        Returns:
            report: Dictionary with generated report and metadata
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        self.model.eval()

        # Encode input
        src_ids = self.tokenizer.encode(threat_indicators, max_len=100)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Generate
        generated_ids, attention_weights = self.model.generate(
            src_tensor,
            max_len=max_len,
            temperature=temperature
        )

        # Decode output
        report_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())

        return {
            "input": threat_indicators,
            "generated_report": report_text,
            "num_tokens": len(generated_ids[0]),
            "attention_available": len(attention_weights) > 0
        }

    def analyze_threat_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze security-related text

        Args:
            text: Security text to analyze

        Returns:
            analysis: Dictionary with analysis results
        """
        tokens = self.tokenizer.tokenize(text)

        # Identify security terms
        security_terms_found = [t for t in tokens if any(st.lower().replace(" ", "_") in t for st in self.tokenizer.security_tokens)]

        # Simple sentiment/threat level
        threat_keywords = ["critical", "severe", "high", "exploit", "vulnerability", "attack", "breach"]
        threat_level = sum(1 for t in tokens if t in threat_keywords) / max(len(tokens), 1)

        return {
            "num_tokens": len(tokens),
            "security_terms": list(set(security_terms_found)),
            "threat_level_score": min(threat_level * 10, 10),
            "tokens": tokens[:20]  # First 20 tokens
        }


# Example usage
if __name__ == "__main__":
    # Initialize NLP engine
    nlp_engine = CustomNLPEngine(vocab_size=5000)

    # Sample security texts for vocabulary
    sample_texts = [
        "CVE-2024-1234 critical SQL injection vulnerability discovered in web application",
        "APT28 threat actor using zero-day exploit for lateral movement",
        "Ransomware attack detected with double extortion tactics",
        "Privilege escalation through buffer overflow in system service"
    ]

    # Build vocabulary
    nlp_engine.train_vocabulary(sample_texts)
    nlp_engine.initialize_model()

    # Analyze threat text
    analysis = nlp_engine.analyze_threat_text(sample_texts[0])
    print("Threat Analysis:")
    print(f"  Tokens: {analysis['num_tokens']}")
    print(f"  Security Terms: {analysis['security_terms']}")
    print(f"  Threat Score: {analysis['threat_level_score']:.2f}/10")

    print("\n✅ Custom NLP Engine test passed!")
