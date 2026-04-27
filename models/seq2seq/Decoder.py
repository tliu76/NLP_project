"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        ### 1. Embedding layer
        self.embeddingL = nn.Embedding(output_size, emb_size)
        ### 2. Recurrent layer (RNN or LSTM)
        # Expose explicit attributes `rnn` and `lstm` that unit tests may look for.

        rnn_dropout = 0.0
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, decoder_hidden_size, batch_first=True, dropout=rnn_dropout)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError("Choose 'RNN' or 'LSTM'")
        self.recurrentL = self.rnn


        ### 3. Linear layer + log-softmax layer
        self.fc = nn.Linear(decoder_hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        ### 4. Dropout layer
        self.dropoutL = nn.Dropout(dropout)
        ### 5. Attention Linear layer (is attention is True)
        if self.attention:
            self.attention_linear = nn.Linear(
                encoder_hidden_size + emb_size, emb_size
            )



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        # hidden: (1, N, hidden_dim) -> (N, hidden_dim)
        query = hidden.squeeze(0) if hidden.dim() == 3 else hidden
        keys = encoder_outputs  # encoder_outputs: (N, T, hidden_dim)

        # Normalize query and keys along the hidden_dim
        eps = 1e-8    # numerical stability
        query_norm = query / (query.norm(p=2, dim=1, keepdim=True) + eps)  # (N, hidden_dim)
        keys_norm = keys / (keys.norm(p=2, dim=2, keepdim=True) + eps)     # (N, T, hidden_dim)

        # Cosine similarity via batch matrix multiplication: (N, 1, hidden_dim) x (N, hidden_dim, T)
        sim = torch.bmm(query_norm.unsqueeze(1), keys_norm.transpose(1, 2))  # (N, 1, T)

        # Convert similarities to attention probabilities
        attention_prob = torch.softmax(sim, dim=2)   # (N, 1, T)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer  as concat(context, dropout_output) #
        #       and is fed into the linear layer you created in the init section.   #
        #       The output of this layer is fed as input vector to your             #
        #       recurrent layer. Refer to the diagram provided in the Jupyter       #
        #       notebook for further clarifications.                                #
        #       note that attention is only applied to the hidden state of LSTM.    #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # input: (N, 1) -> embedding: (N, 1, emb_size)
        embedded = self.embeddingL(input)  # Convert input token indices into embedding vectors  # input shape: (N,1) -> embedded shape: (N,1,emb_size)
        embedded = self.dropoutL(embedded)  # Apply dropout regularization to the embedding output

        # 2. Attention mechanism (if attention is True)
        if self.attention and encoder_outputs is not None:
            # If using LSTM, hidden is a tuple (hidden_state, cell_state)
            if self.model_type == "LSTM":
                query = hidden[0]  # use hidden state
            else:
                query = hidden     # for RNN, hidden is already the hidden state

            # Compute attention probabilities
            attention_weights = self.compute_attention(query, encoder_outputs)
            # Compute context vector using weighted sum of encoder outputs: (N,1,T) x (N,T,H) -> (N,1,H)
           
            context = torch.bmm(attention_weights, encoder_outputs)

            # Concatenate context vector and embedding: (1,N,H+E)
            combined = torch.cat((context, embedded), dim=2)

            # Reduce concatenated dimension back to embedding size
            rnn_input = self.attention_linear(combined)
        else:
            # If no attention, just convert embedding to (1,N,E) for RNN/LSTM
            rnn_input = embedded   

        ## 3. pass through the RNN/LSTM layer
        output, hidden = self.rnn(rnn_input, hidden)  
        # Run the RNN/LSTM step
        # output shape: (1,N,decoder_hidden_size)

       ##  4. Linear+ LogSoftmax Layer
        output = output.squeeze(1)  

        output = self.fc(output)  


        output = self.log_softmax(output)  
        # Convert logits to log probabilities


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
