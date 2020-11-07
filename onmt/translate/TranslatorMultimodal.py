import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io

import numpy as np
import scipy.misc
from skimage import transform, filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

class TranslatorMultimodal(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0,
                 test_img_feats=None, multimodal_model_type=None, opt=None):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.test_img_feats = test_img_feats
        self.multimodal_model_type = multimodal_model_type
        self.opt = opt

        assert(not test_img_feats is None),\
                'Please provide file with test image features.'
        assert(not multimodal_model_type is None),\
                'Please provide the multimodal model type name.'

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        def get_imglist():
            imglist = []
            num = 0
            with open(self.opt.path_to_imglist, 'r') as f:
                for img in f:
                    img_path = "%s/%s" % (self.opt.path_to_raw_imgs, img.strip())
                    imglist.append(img_path)
                    num += 1
            print(("load %d raw img from %s") % (num, self.opt.path_to_imglist))
            return imglist
        self.imglist = get_imglist()

    def get_heatmap(self, sent_idx, attn):
        def get_blend_map(img, att_map, blur=True, overlap=True):
            att_map -= att_map.min()
            if att_map.max() > 0:
                att_map /= att_map.max()
            att_map = transform.resize(att_map, (img.shape[:2]), order=3, mode='nearest')
            if blur:
                att_map = filters.gaussian_filter(att_map, 0.02 * max(img.shape[:2]))
                att_map -= att_map.min()
                att_map /= att_map.max()
            cmap = plt.get_cmap('jet')
            att_map_v = cmap(att_map)
            att_map_v = np.delete(att_map_v, 3, 2)
            if overlap:
                att_map = 1 * (1 - att_map ** 0.7).reshape(att_map.shape + (1,)) * img + (att_map ** 0.7).reshape(
                    att_map.shape + (1,)) * att_map_v
            return att_map
        # with open(self.imglist[sent_idx], 'rb') as f:
        #     with Image.open(f) as img:
        #         img = img.convert('RGB')
        img = scipy.misc.imread(self.imglist[sent_idx])
        img = scipy.misc.imresize(img, (300, 300), interp='bicubic')
        attn = attn.data.numpy()
        attn = np.squeeze(attn).reshape(attn.shape[-1], attn.shape[-2])
        for i in range(attn.shape[0]):
            heatmap = get_blend_map(img/255., attn[i][-49:].reshape([7, 7]))
            plt.imshow(heatmap)
            plt.savefig('heatmap.0/' + str(sent_idx) + '_heatmap_word' + str(i), bbox_inches='tight')
            # plt.save('heatmap/' + str(sent_idx) + '_heatmap_word' + str(i))
            # break

        # attn = attn.data.numpy()
        # attn = np.squeeze(attn).reshape(attn.shape[-1], attn.shape[-2])
        # for i in range(attn.shape[0]):
        #     heatmap = get_blend_map(img / 255., attn[i][-49:].reshape([7, 7]))
        #     plt.imshow(heatmap)
        #     plt.savefig('heatmap/' + str(sent_idx) + '_heatmap_word' + str(i), bbox_inches='tight')

    def translate_batch(self, batch, data, sent_idx):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           sent_idx: the sentence idxs mapping to the image features

        Todo:
           Shouldn't need the original dataset.
        """

        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy( self.test_img_feats[sent_idx] )
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)
        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # project image features
        img_proj = self.model.encoder_images( img_feats )

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size # here batch_size = 1
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        #enc_states, context = self.model.encoder(src, src_lengths)
        #dec_states = self.model.decoder.init_decoder_state(
        #                                src, context, enc_states)
        if self.multimodal_model_type == 'imge':
            # create initial hidden state differently for GRU/LSTM
            if self.model._evaluate_is_tuple_hidden(src, src_lengths):
                enc_init_state = (img_proj, img_proj)
            else:
                enc_init_state = img_proj
            # initialise encoder with image features
            enc_states, context = self.model.encoder(src, src_lengths, enc_init_state)
            # traditional decoder
            dec_states = self.model.decoder.init_decoder_state(src, context, enc_states)
        elif self.multimodal_model_type == 'imgd':
            # traditional encoder
            enc_states, context = self.model.encoder(src, src_lengths)
            # combine encoder final hidden state with image features
            enc_init_state = self.model._combine_enc_state_img_proj(enc_states, img_proj)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                                            src, context, enc_init_state)
        elif self.multimodal_model_type == 'imgw':
            #enc_states, context = self.model.encoder(src, src_lengths, img_proj)
            # use image features as words in the encoder
            enc_states, context = self.model.encoder(src, img_feats=img_proj, lengths=src_lengths)
            # update the lengths variable with the new source lengths after incorporating image feats
            src_lengths = self.model.encoder.updated_lengths
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(src, context, enc_states)
        elif self.multimodal_model_type == 'src+img':
            # traditional encoder
            enc_states, context = self.model.encoder(src, src_lengths)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(src,
                    context, img_proj, enc_states)
        elif self.multimodal_model_type == 'graphtransformer':
            # traditional encoder
            input_mm, context, attn = self.model.encoder.forward_mm(src, img_proj, src_lengths) # attn 1x65x16
            # initialise decoder
            # here need to transpose because of beam search
            # sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            #self.get_heatmap(sent_idx, attn)
            dec_states = self.model.decoder.init_decoder_state(input_mm.transpose(0, 1),
                    context, None)
        else:
            raise Exception("Multi-modal model not implemented: %s"%self.multimodal_model_type)

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        # image features are in (batch x len x feats),
        # but rvar() function expects (len x batch x feats)
        img_proj = rvar(img_proj.transpose(0,1).data)
        # return it back to (batch x len x feats)
        img_proj = img_proj.transpose(0,1)
        context_lengths = src_lengths.repeat(beam_size)
        # so after transpose here use repeat_beam_size_times instead of repeat_beam_size_times_mm
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # def var(a): return Variable(a, volatile=True)
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2) # (1, 5, 1)

            # Run one step.
            if self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'graphtransformer']:
                dec_out, dec_states, attn = self.model.decoder(
                    inp, context, dec_states, context_lengths=context_lengths) # dec_out (1, 5, 512) (batch, beam, dim)
            elif self.multimodal_model_type == 'src+img':
                dec_out, dec_out_imgs, dec_states, attn = self.model.decoder(
                        inp, context, img_proj, dec_states,
                        context_lengths=context_lengths)
            else:
                raise Exception("Multi-modal model type not implemented: %s"%(
                    self.multimodal_model_type))

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                # define in ModelConstructor: Sequential(Linear, LogSoftmax)
                out = self.model.generator.forward(dec_out).data # (beam, vocab_size)
                out = unbottle(out) # (5, 1, 18605)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j], # (5, 18605)
                    unbottle(attn["std"]).data[:, j, :context_lengths[j]]) # attn (beam, batch-1, tgtlen)
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data, sent_idx):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        #enc_states, context = self.model.encoder(src, src_lengths)
        #dec_states = self.model.decoder.init_decoder_state(src,
        #                                                   context, enc_states)
        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy( self.test_img_feats[sent_idx] )
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)
        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # project image features
        img_proj = self.model.encoder_images( img_feats )
        if self.multimodal_model_type == 'imge':
            # initialise encoder with image features
            enc_states, context = self.model.encoder(src, src_lengths, img_proj)
            # traditional decoder
            dec_states = self.model.decoder.init_decoder_state(src, context, enc_states)
        elif self.multimodal_model_type == 'imgd':
            # traditional encoder
            enc_states, context = self.model.encoder(src, src_lengths)
            # combine encoder final hidden state with image features
            enc_init_state = self.model._combine_enc_state_img_proj(enc_states, img_proj)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                                            src, context, enc_init_state)
        elif self.multimodal_model_type == 'imgw':
            #enc_states, context = self.model.encoder(src, src_lengths, img_proj)
            # use image features as words in the encoder
            enc_states, context = self.model.encoder(src, img_feats=img_proj, lengths=src_lengths)
            # update the lengths variable with the new source lengths after incorporating image feats
            src_lengths = self.model.encoder.updated_lengths
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(src, context, enc_states)
        elif self.multimodal_model_type == 'src+img':
            # traditional encoder
            enc_states, context = self.model.encoder(src, src_lengths)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(src,
                    context, img_proj, enc_states)
        elif self.multimodal_model_type == 'graphtransformer':
            # traditional encoder
            input_mm, context = self.model.encoder.forward_mm(src, img_proj, src_lengths)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(input_mm,
                    context, None)
        else:
            raise Exception("Multi-modal model not implemented: %s"%self.multimodal_model_type)


        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        if self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'graphtransformer']:
            dec_out, dec_states, attn = self.model.decoder(
                tgt_in, context, dec_states, context_lengths=src_lengths)
        elif self.multimodal_model_type == 'src+img':
            dec_out, dec_out_imgs, dec_states, attn = self.model.decoder(
                    tgt_in, context, img_proj, dec_states,
                    context_lengths=src_lengths)
        else:
            raise Exception("Multi-modal odel type not implemented: %s"%(
                self.multimodal_model_type))

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
