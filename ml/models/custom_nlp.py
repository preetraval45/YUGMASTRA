"""
Custom NLP Engine for YUGMĀSTRA

Built from scratch for:
- Security report generation
- Attack description to natural language
- Rule explanation
- Threat intelligence summarization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re


class TextTokenizer:
    """Custom tokenizer for cybersecurity text"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq = {}

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Tokenize and count words
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1

        # Keep most frequent words
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.vocab_size - len(self.word2idx)]

        # Build vocab dictionaries
        for idx, (word, _) in enumerate(sorted_words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

        if max_len:
            if len(ids) < max_len:
                ids += [self.word2idx['<PAD>']] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [
            self.idx2word.get(idx, '<UNK>')
            for idx in ids
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<START>'], self.word2idx['<END>']]
        ]
        return ' '.join(tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (lowercase, split on non-alphanumeric)"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence encoding"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        return output


class AttentionLayer(nn.Module):
    """Attention mechanism for NLP"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # encoder_outputs: (batch, seq_len, hidden*2)
        # hidden: (batch, hidden)

        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate and compute attention scores
        energy = torch.tanh(self.attention(torch.cat([encoder_outputs, hidden], dim=2)))
        attention_scores = self.v(energy).squeeze(2)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        return context.squeeze(1), attention_weights


class ReportGeneratorLSTM(nn.Module):
    """
    LSTM-based report generator

    Generates natural language security reports from structured data
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = BidirectionalLSTM(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout
        )

        # Attention
        self.attention = AttentionLayer(hidden_dim)

        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        # Encode source
        encoder_outputs = self.encoder(src)

        batch_size = tgt.size(0)
        max_len = tgt.size(1)

        # Initialize decoder hidden state
        decoder_hidden = torch.zeros(
            2, batch_size, self.hidden_dim,
            device=src.device
        )
        decoder_cell = torch.zeros(
            2, batch_size, self.hidden_dim,
            device=src.device
        )

        outputs = []

        # Decode step by step
        for t in range(max_len):
            decoder_input = tgt[:, t].unsqueeze(1)

            # Get context from attention
            context, _ = self.attention(
                encoder_outputs,
                decoder_hidden[-1]
            )

            # Embed decoder input
            embedded = self.decoder_embedding(decoder_input)

            # Concatenate with context
            decoder_input_with_context = torch.cat(
                [embedded, context.unsqueeze(1)],
                dim=2
            )

            # Decode
            output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input_with_context,
                (decoder_hidden, decoder_cell)
            )

            # Project to vocabulary
            prediction = self.output_projection(output.squeeze(1))
            outputs.append(prediction)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        start_token: int,
        end_token: int,
        temperature: float = 1.0
    ) -> List[int]:
        """Generate text autoregressively"""
        self.eval()
        device = src.device

        # Encode
        encoder_outputs = self.encoder(src)

        # Initialize decoder
        decoder_hidden = torch.zeros(2, 1, self.hidden_dim, device=device)
        decoder_cell = torch.zeros(2, 1, self.hidden_dim, device=device)

        generated = [start_token]

        with torch.no_grad():
            for _ in range(max_len):
                decoder_input = torch.tensor([[generated[-1]]], device=device)

                # Attention
                context, _ = self.attention(encoder_outputs, decoder_hidden[-1])

                # Embed and decode
                embedded = self.decoder_embedding(decoder_input)
                decoder_input_with_context = torch.cat(
                    [embedded, context.unsqueeze(1)],
                    dim=2
                )

                output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    decoder_input_with_context,
                    (decoder_hidden, decoder_cell)
                )

                # Predict next token
                logits = self.output_projection(output.squeeze(1)) / temperature
                probs = F.softmax(logits, dim=1)
                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                if next_token == end_token:
                    break

        return generated


class SecurityReportGenerator:
    """
    High-level interface for generating security reports
    """

    def __init__(self, model: ReportGeneratorLSTM, tokenizer: TextTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_attack_report(
        self,
        attack_data: Dict[str, Any]
    ) -> str:
        """
        Generate natural language report from attack data

        Args:
            attack_data: Dictionary with attack information
                {
                    'type': 'phishing',
                    'target': 'web_server',
                    'success': True,
                    'techniques': ['T1566', 'T1059'],
                    'impact': 'data_exfiltration'
                }

        Returns:
            report: Natural language report
        """
        # Convert attack data to input sequence
        input_text = self._attack_data_to_text(attack_data)
        input_ids = self.tokenizer.encode(input_text, max_len=128)
        input_tensor = torch.tensor([input_ids])

        # Generate report
        generated_ids = self.model.generate(
            src=input_tensor,
            max_len=200,
            start_token=self.tokenizer.word2idx['<START>'],
            end_token=self.tokenizer.word2idx['<END>'],
            temperature=0.7
        )

        # Decode to text
        report = self.tokenizer.decode(generated_ids)

        return self._format_report(report)

    def generate_defense_summary(
        self,
        defense_data: Dict[str, Any]
    ) -> str:
        """Generate summary of defensive actions"""
        input_text = self._defense_data_to_text(defense_data)
        input_ids = self.tokenizer.encode(input_text, max_len=128)
        input_tensor = torch.tensor([input_ids])

        generated_ids = self.model.generate(
            src=input_tensor,
            max_len=150,
            start_token=self.tokenizer.word2idx['<START>'],
            end_token=self.tokenizer.word2idx['<END>'],
            temperature=0.7
        )

        summary = self.tokenizer.decode(generated_ids)
        return self._format_report(summary)

    def explain_detection_rule(
        self,
        rule_data: Dict[str, Any]
    ) -> str:
        """Generate explanation for detection rule"""
        input_text = f"rule {rule_data.get('name', 'unknown')} detects {rule_data.get('description', '')}"
        input_ids = self.tokenizer.encode(input_text, max_len=64)
        input_tensor = torch.tensor([input_ids])

        generated_ids = self.model.generate(
            src=input_tensor,
            max_len=100,
            start_token=self.tokenizer.word2idx['<START>'],
            end_token=self.tokenizer.word2idx['<END>'],
            temperature=0.5
        )

        explanation = self.tokenizer.decode(generated_ids)
        return explanation

    def _attack_data_to_text(self, attack_data: Dict[str, Any]) -> str:
        """Convert attack data dict to input text"""
        parts = []
        if 'type' in attack_data:
            parts.append(f"attack type {attack_data['type']}")
        if 'target' in attack_data:
            parts.append(f"target {attack_data['target']}")
        if 'success' in attack_data:
            parts.append(f"{'successful' if attack_data['success'] else 'failed'}")
        if 'techniques' in attack_data:
            parts.append(f"techniques {' '.join(attack_data['techniques'])}")
        if 'impact' in attack_data:
            parts.append(f"impact {attack_data['impact']}")

        return ' '.join(parts)

    def _defense_data_to_text(self, defense_data: Dict[str, Any]) -> str:
        """Convert defense data dict to input text"""
        parts = []
        if 'action' in defense_data:
            parts.append(f"defense action {defense_data['action']}")
        if 'detected' in defense_data:
            parts.append(f"detected {defense_data['detected']}")
        if 'blocked' in defense_data:
            parts.append(f"blocked {defense_data['blocked']}")

        return ' '.join(parts)

    def _format_report(self, text: str) -> str:
        """Format and clean generated report"""
        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else ""

        # Add period if missing
        if text and text[-1] not in '.!?':
            text += '.'

        return text
