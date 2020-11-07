from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq
import sys

class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, input, context, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(
            input, context, state, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all: # def _all(self): return (self.previous_input, self.src)
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class ImageGlobalFeaturesProjector(nn.Module):
    """
        Project global image features using a 2-layer multi-layer perceptron.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): number of decoder layers.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ImageGlobalFeaturesProjector, self).__init__()
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.outdim = outdim
        self.dropout = dropout
        
        layers = []
        layers.append( nn.Linear(nfeats, nfeats) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        # final layers projects from nfeats to decoder rnn hidden state size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        #print( "out.size(): ", out.size() )
        if self.num_layers>1:
            out = out.unsqueeze(0)
            out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
            #print( "out.size(): ", out.size() )
        return out


class View(nn.Module):
    """Helper class to be used inside Sequential object to reshape Variables"""
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class ImageLocalFeaturesProjector(nn.Module):
    """
        Reshape local image features.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): 1.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ImageLocalFeaturesProjector, self).__init__()
        assert(num_layers==1), \
                'num_layers must be equal to 1 !'
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.dropout = dropout
        
        layers = []
        # reshape input
        layers.append( View(-1, 7*7, nfeats) )
        # linear projection from feats to rnn size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        #print( "out.size(): ", out.size() )
        #if self.num_layers>1:
        #    out = out.unsqueeze(0)
        #    out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
        #    #print( "out.size(): ", out.size() )
        return out


class RNNEncoderImageAsWord(EncoderBase):
    """ The RNN encoder that uses image features as words. """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(RNNEncoderImageAsWord, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, img_feats, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        # concatenate image features as source words
        emb = torch.cat((img_feats[0].unsqueeze(0), emb, img_feats[1].unsqueeze(0)), 0)
        # length after concatenating two new words to source
        assert(emb.size(0) == s_len+2), 'Dimensions do not match.'

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # adjust lengths variable to include the two new "image-words"
            lengths+=2
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        # store updated lengths variable (could be needed in `translate_mm.py`)
        self.updated_lengths = lengths

        return hidden_t, outputs


class RNNDecoderBaseDoublyAttentive(nn.Module):
    """
    Base recurrent doubly-attentive-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTSrcImgModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBaseDoublyAttentive, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )
        self.attn_img = onmt.modules.GlobalAttention(
            hidden_size,
            coverage=False, # coverage not yet implemented for visual attention
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, input, context, context_img, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            context_img (`FloatTensor`): vectors from the image
                 `[img_feats_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # transpose image feats from (batch x len x feats) to (len x batch x feats)
        context_img = context_img.transpose(0,1)

        # Args Check
        assert isinstance(state, RNNDecoderStateDoublyAttentive)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        contxt_img_len, contxt_img_batch, _ = context_img.size()
        aeq(input_batch, contxt_batch)
        aeq(input_batch, contxt_img_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        #hidden, outputs, attns, coverage = self._run_forward_pass(
        hidden, outputs, outputs_img, attns, coverage, coverage_img = \
                self._run_forward_pass(input, context, context_img,
                        state, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        final_output_img = outputs_img[-1]
        #state.update_state(hidden, final_output.unsqueeze(0),
        #                   coverage.unsqueeze(0)
        #                   if coverage is not None else None)
        state.update_state(hidden,
                           final_output.unsqueeze(0),
                           final_output_img.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None,
                           coverage_img.unsqueeze(0)
                           if coverage_img is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        outputs_img = torch.stack(outputs_img)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, outputs_img, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, context_img, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            #return RNNDecoderState(context, self.hidden_size,
            #                       tuple([self._fix_enc_hidden(enc_hidden[i])
            #                             for i in range(len(enc_hidden))]))
            return RNNDecoderStateDoublyAttentive(context, context_img,
                                   self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            #return RNNDecoderState(context, self.hidden_size,
            #                       self._fix_enc_hidden(enc_hidden))
            return RNNDecoderStateDoublyAttentive(context, context_img,
                                   self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoderDoublyAttentive(RNNDecoderBaseDoublyAttentive):
    """
    Standard fully batched RNN decoder with two attention heads.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, input, context, context_img, state,
            context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            context_img (FloatTensor): output(tensor sequence) from the pretrained
                        CNN of size (img_feats_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of outputs (source-language)
                                     of every time step from the decoder.
            outputs_img ([FloatTensor]): an array of outputs (image features)
                                         of every time step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        outputs_img = []
        attns = {"std": [], "std_img": []}
        coverage = None
        coverage_img = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores
        attn_img_outputs, attn_img_scores = self.attn_img(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context_img.transpose(0, 1),              # (contxt_img_len, batch, d)
            context_lengths=None
        )
        attns["std_img"] = attn_img_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # no context gate on image attention
        outputs_img = self.dropout(attn_img_outputs)

        # Return result.
        return hidden, outputs, outputs_img, attns, coverage, coverage_img

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoderDoublyAttentive(RNNDecoderBaseDoublyAttentive):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, input, context, context_img, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        outputs_img = []
        attns = {"std": [], "std_img": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None
        # npt using coverage in visual attention
        coverage_img = None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            attn_output_img, attn_img = self.attn_img(
                rnn_output,
                context_img.transpose(0, 1),
                context_lengths=None)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            output_img = self.dropout(attn_output_img)
            outputs += [output]
            outputs_img += [output_img]
            attns["std"] += [attn]
            attns["std_img"] += [attn_img]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, outputs, outputs_img, attns, coverage, coverage_img

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size




class NMTImgDModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model
    with image features used to initialise the decoder.
    """
    def __init__(self, encoder, decoder, encoder_images, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            encoder_images(Encoder): the image encoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTImgDModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_images = encoder_images

    def _combine_enc_state_img_proj(self, enc_hidden, img_proj):
        """
        Args:
            enc_hidden(tuple or DecoderState):
                                    Tuple containing hidden state and cell
                                    (h,c) in case of an LSTM, or a hidden state
                                    (DecoderState) in case of GRU cell.
            img_proj(Variable):     Variable containing projected image features.
            
            Returns:
                Variable with DecoderState combined with image features.
        """
        enc_init_state = []
        if isinstance(enc_hidden, tuple):
            for e in enc_hidden:
                enc_init_state.append(e + img_proj)
            enc_init_state = tuple(enc_init_state)
        else:
            enc_init_state = enc_hidden + img_proj
        return enc_init_state


    def forward(self, src, tgt, lengths, img_feats, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            img_feats(FloatTensor): image features of size (batch x nfeats).
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        # project image features
        img_proj = self.encoder_images(img_feats)

        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_init_state = self._combine_enc_state_img_proj(enc_hidden, img_proj)
        #enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        enc_state = self.decoder.init_decoder_state(src, context, enc_init_state)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class NMTImgEModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model
    with image features used to initialise the encoder.
    """
    def __init__(self, encoder, decoder, encoder_images, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            encoder_images(Encoder): the image encoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTImgEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_images = encoder_images

        # workaround
        self.is_tuple_hidden = None

    def _evaluate_is_tuple_hidden(self, src, lengths):
        # TODO: workaround to access the dimensions of the encoder hidden state.
        # do it without running the forward pass on the encoder!
        if self.is_tuple_hidden is None:
            enc_hidden, context = self.encoder(src, lengths)
            if isinstance(enc_hidden, tuple):
                self.is_tuple_hidden = True
            else:
                self.is_tuple_hidden = False
        ret = self.is_tuple_hidden
        return ret

    def forward(self, src, tgt, lengths, img_feats, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            img_feats(FloatTensor): image features of size (batch x nfeats).
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        # project image features
        img_proj = self.encoder_images(img_feats)

        tgt = tgt[:-1]  # exclude last target from inputs

        # create initial hidden state differently for GRU/LSTM
        if self._evaluate_is_tuple_hidden(src, lengths):
            enc_init_state = (img_proj, img_proj)
        else:
            enc_init_state = img_proj

        enc_hidden, context = self.encoder(src, lengths, enc_init_state)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class NMTImgWModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model
    with image features used to initialise the encoder.
    """
    def __init__(self, encoder, decoder, encoder_images, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            encoder_images(Encoder): the image encoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTImgWModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_images = encoder_images

    def forward(self, src, tgt, lengths, img_feats, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            img_feats(FloatTensor): image features of size (batch x nfeats).
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        # project image features
        img_proj = self.encoder_images(img_feats)

        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, img_feats=img_proj, lengths=lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class NMTSrcImgModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a doubly-attentive encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, encoder_images, multigpu=False):
        self.multigpu = multigpu
        super(NMTSrcImgModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_images = encoder_images

    def forward(self, src, tgt, lengths, img_feats, dec_state=None):
        """Forward propagate a `src`, `img_feats` and `tgt` tuple for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            img_feats(FloatTensor): image features of size (len x batch x nfeats).
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        img_proj = self.encoder_images( img_feats ) # (49, 512)

        # 在这里 把img_proj加到transformer里
        enc_hidden, context = self.encoder.forward_mm(src, img_proj, lengths)

        # project/transform local image features into the expected structure/shape
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, attns, dec_state


class GraphTransformer(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a doubly-attentive encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, encoder_images, multigpu=False):
        self.multigpu = multigpu
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_images = encoder_images

    def forward(self, src, tgt, lengths, img_feats, dec_state=None):
        """Forward propagate a `src`, `img_feats` and `tgt` tuple for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            img_feats(FloatTensor): image features of size (len x batch x nfeats).
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        img_proj = self.encoder_images( img_feats ) # (49, 512)

        # 在这里 把img_proj加到transformer里
        input_mm, context = self.encoder.forward_mm(src, img_proj, lengths)

        # project/transform local image features into the expected structure/shape
        # here context and enc_hidden doesnt be used
        # img+src is our input
        enc_state = self.decoder.init_decoder_state(input_mm.transpose(0, 1), context, enc_hidden=None)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class RNNDecoderStateDoublyAttentive(DecoderState):
    def __init__(self, context, context_img, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            context_img (FloatTensor): output from the pretrained CNN of size
                                   img_len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None
        self.coverage_img = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.input_feed_img = Variable(context_img.data.new(*h_size).zero_(),
                                       requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed, self.input_feed_img)

    def update_state(self, rnnstate, input_feed, input_feed_img,
            coverage, coverage_img):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.input_feed_img = input_feed_img
        self.coverage = coverage
        self.coverage_img = coverage_img

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        #self.hidden = tuple(vars[:-1])
        #self.input_feed = vars[-1]
        self.hidden = tuple(vars[:-2])
        self.input_feed = vars[-2]
        self.input_feed_img = vars[-1]
