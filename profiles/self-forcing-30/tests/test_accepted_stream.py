import tempfile
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from cfexp.accepted_stream import AcceptedStream
from test_correction_state import Fixture


class FakeFrame:
    def __init__(self, index):
        self.index = index
        self.shape = (9 if index == 0 else 12, 3, 480, 832)


class Decoder:
    evidence = {'native': 'test fixture'}
    def __init__(self):
        self.index = 0
        self.clears = 0
    def clear(self):
        self.index = 0
        self.clears += 1
    def decode(self, latent):
        result = FakeFrame(self.index)
        self.index += 1
        return result


class Writer:
    def __init__(self, *a, **kw):
        self.indices = []
        self.expected = kw['expected_frames']
    def push(self, pixels):
        self.indices.append(pixels.index)
    def finish(self):
        if self.indices != list(range((self.expected + 3) // 12)):
            raise ValueError('incorrect frame order')
        return {'input_rgb_tchw_sha256': 'a'*64, 'path': 'test.mp4'}
    def abort(self):
        pass


class AcceptedStreamTests(unittest.TestCase):
    def test_paused_full_horizon_queue_does_not_deadlock_and_finish_drains_it(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();history=Fixture(root).baseline(30);decoder=Decoder()
            stream=AcceptedStream(root,history=history.prefix_before(0),output_path='test.mp4',
                chunk_count=30,decoder=decoder,gpu_lock=threading.RLock(),priority=True,
                writer_factory=Writer,load_latent=lambda p:p.read_bytes())
            try:
                stream.set_render_allowed(False)
                for i in range(1,31):stream.append(history.prefix_before(i))
                self.assertEqual(decoder.index,0)
                self.assertEqual(stream.queue.maxsize,30)
                result=stream.finish(history)
                self.assertTrue(result['role_aware_priority'])
                self.assertEqual(result['latent_descriptor_queue_bound'],30)
                self.assertEqual(len(result['chunks']),30)
                self.assertTrue(result['decoder_and_encoder_joined'])
            finally:stream.abort()

    def test_long_horizons_keep_frame_geometry_and_flush_all_descriptors(self):
        for chunks, frames in ((60, 717), (120, 1437)):
            with self.subTest(chunks=chunks), tempfile.TemporaryDirectory() as raw:
                root = Path(raw).resolve()
                history = Fixture(root).baseline(chunks)
                stream = AcceptedStream(root, history=history.prefix_before(0),
                    output_path='test.mp4', chunk_count=chunks, decoder=Decoder(),
                    gpu_lock=threading.RLock(), priority=True, writer_factory=Writer,
                    load_latent=lambda path: path.read_bytes())
                try:
                    self.assertEqual(stream.writer.expected, frames)
                    for index in range(1, chunks + 1):
                        stream.append(history.prefix_before(index))
                    result = stream.finish(history)
                    self.assertEqual(result['pixels']['shape'][0], frames)
                    self.assertEqual(result['chunks'][-1]['frame_start'] + 12, frames)
                    self.assertEqual(result['latent_descriptor_queue_bound'], chunks)
                finally:
                    stream.abort()

    def test_abort_wakes_a_paused_decoder_without_publishing(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();history=Fixture(root).baseline(30)
            stream=AcceptedStream(root,history=history.prefix_before(0),output_path='test.mp4',
                chunk_count=30,decoder=Decoder(),gpu_lock=threading.RLock(),priority=True,
                writer_factory=Writer,load_latent=lambda p:p.read_bytes())
            stream.set_render_allowed(False);stream.append(history.prefix_before(1));stream.abort()
            self.assertFalse(stream.thread.is_alive())
            self.assertFalse((root/'test.mp4').exists())

    def test_client_pauses_once_per_target_burst_and_resumes_before_draft(self):
        from types import SimpleNamespace
        from cfexp.accepted_stream_client import StreamDecoderProxy
        from cfexp.streaming_vae_probe import VAE_SHA,WEIGHTS_SHA
        roles=[]
        def request(action,fields):
            if action=='stream_prepare':
                return dict(native_vae_sha256=VAE_SHA,checkpoint_sha256=WEIGHTS_SHA,separate_causal_decoder_state=True)
            roles.append(fields['generation_role'])
            return dict(allowed=fields['generation_role']!='target',generation_role=fields['generation_role'],renderer_worker_role='target')
        with patch.dict('os.environ',{'CFEXP_RENDER_TARGET_PRIORITY':'1'}):
            proxy=StreamDecoderProxy(SimpleNamespace(role='draft'),SimpleNamespace(role='target',_request=request),Path('/fixture'))
            for role in ['draft','draft','target','target','target','target','draft','draft']:
                proxy.before_generate(role)
        self.assertEqual(roles,['target','draft'])

    def test_separate_worker_entrypoint_preserves_original_class_and_launcher(self):
        from cfexp import correction_worker as base
        from cfexp import accepted_stream_worker as stream
        original=base.CorrectionWorker
        with patch.dict('os.environ',{'CFEXP_ACCEPTED_VIDEO_STREAM':'1'}):
            with patch('cfexp.accepted_stream_worker.subprocess.Popen') as launch:
                stream.process_factory(['python','-m','cfexp.correction_worker','--role','target'],cwd='/tmp',env={'test':'fixture'})
                self.assertEqual(launch.call_args.args[0],['python','-m','cfexp.accepted_stream_worker','--role','target'])
                self.assertEqual(launch.call_args.kwargs,{'cwd':'/tmp','env':{'test':'fixture'}})
                with self.assertRaises(ValueError):stream.process_factory(['python','-m','arbitrary'])
            def run(argv):
                self.assertIs(base.CorrectionWorker,stream.StreamingCorrectionWorker)
                return 0
            with patch.object(base,'main',side_effect=run):
                self.assertEqual(stream.main(['--test-fixture']),0)
        self.assertIs(base.CorrectionWorker,original)

    def test_worker_protocol_is_opt_in_and_reuses_the_prepared_decoder(self):
        from types import SimpleNamespace
        from cfexp.accepted_stream_worker import handle
        from cfexp.correction_worker import WorkerProtocolError
        from cfexp.streaming_vae_probe import VAE_SHA, WEIGHTS_SHA
        with tempfile.TemporaryDirectory() as raw:
            worker=SimpleNamespace(_chunk_count=30,run_dir=Path(raw).resolve(),
                _render_gpu_lock=threading.RLock(),
                _COMMON_KEYS={'schema','request_id','role','action'})
            request=dict(schema='test',request_id='test',role='target',action='stream_prepare')
            with patch.dict('os.environ',{},clear=True):
                with self.assertRaises(WorkerProtocolError): handle(worker,request)
            decoder=SimpleNamespace(evidence=dict(native_vae_sha256=VAE_SHA,
                checkpoint_sha256=WEIGHTS_SHA,separate_causal_decoder_state=True))
            env={'CFEXP_ACCEPTED_VIDEO_STREAM':'1','CFEXP_DRAFT_KV_REUSE':'1',
                 'CFEXP_VIDEO_ENCODING':'serial_abr_v1'}
            with patch.dict('os.environ',env),patch('cfexp.accepted_stream_worker.NativeStreamDecoder',return_value=decoder) as build:
                self.assertEqual(handle(worker,request),decoder.evidence)
                self.assertEqual(handle(worker,request),decoder.evidence)
                build.assert_called_once()
                with self.assertRaises(WorkerProtocolError): handle(worker,dict(request,extra=True))

    def test_proxy_preserves_strict_video_metadata_and_bound_evidence(self):
        from types import SimpleNamespace
        from cfexp.accepted_stream_client import StreamDecoderProxy
        from cfexp.streaming_vae_probe import VAE_SHA, WEIGHTS_SHA
        import hashlib
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();history=Fixture(root).baseline(30)
            (root/'videos').mkdir();payload=b'encoded-video-fixture';(root/'videos/source.mp4').write_bytes(payload)
            native=dict(native_vae_sha256=VAE_SHA,checkpoint_sha256=WEIGHTS_SHA,separate_causal_decoder_state=True)
            result=dict(history_manifest_sha256=history.manifest_sha256,
                history_head_sha256=history.head_sha256,latent_chunk_count=30,decoder_and_encoder_joined=True,
                chunks=[dict(segment_sha256=s.segment_sha256) for s in history.segments],
                video=dict(path='videos/source.mp4',sha256=hashlib.sha256(payload).hexdigest(),nbytes=len(payload),fps=24,frames=357,
                    input_rgb_tchw_sha256='a'*64,encoding_policy='serial_abr_v1'),
                pixels=dict(layout='TCHW',dtype='uint8',shape=[357,3,480,832],raw_sha256='a'*64,raw_nbytes=357*3*480*832))
            def request(action,fields):
                if action=='stream_prepare':return native
                if action=='stream_append':
                    return dict(history_head_sha256=fields['history']['head_sha256'],latent_chunk_count=fields['history']['next_chunk_index'])
                return result
            proxy=StreamDecoderProxy(SimpleNamespace(role='draft'),SimpleNamespace(_request=request),root)
            for i in range(1,31):proxy.append(history.prefix_before(i))
            value=proxy.decode_video(history=history,output_video_path='videos/alias.mp4')
            self.assertEqual(set(value.video),{'path','sha256','nbytes','fps','frames'})
            self.assertEqual((root/'videos/alias.mp4').read_bytes(),payload)
            self.assertEqual(proxy.last_evidence,result)
            self.assertTrue((root/'provenance/accepted-stream'/(history.manifest_sha256+'.json')).exists())

    def test_complete_accepted_prefix_and_causal_frame_positions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            history = Fixture(root).baseline(30)
            decoder = Decoder()
            stream = AcceptedStream(root, history=history.prefix_before(0), output_path='test.mp4',
                chunk_count=30, decoder=decoder, gpu_lock=threading.RLock(),
                writer_factory=Writer, load_latent=lambda p: p.read_bytes())
            try:
                for i in range(1, 31):
                    stream.append(history.prefix_before(i))
                result = stream.finish(history)
                self.assertTrue(result['decoder_and_encoder_joined'])
                self.assertEqual(result['history_head_sha256'], history.head_sha256)
                self.assertEqual([r['frame_start'] for r in result['chunks']],
                    [0]+[12*i-3 for i in range(1,30)])
                self.assertEqual([r['segment_sha256'] for r in result['chunks']],
                    [s.segment_sha256 for s in history.segments])
                self.assertEqual(result['pixels']['shape'], [357,3,480,832])
                self.assertEqual(decoder.clears, 2)
            finally:
                stream.abort()
            self.assertFalse(stream.thread.is_alive())

    def test_reject_skip_duplicate_branch_switch_and_incomplete_finish(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(); history = Fixture(root).baseline(30)
            stream = AcceptedStream(root, history=history.prefix_before(0), output_path='test.mp4',
                chunk_count=30, decoder=Decoder(), gpu_lock=threading.RLock(),
                writer_factory=Writer, load_latent=lambda p: p.read_bytes())
            try:
                with self.assertRaisesRegex(ValueError, 'extension'):
                    stream.append(history.prefix_before(2))
                stream.append(history.prefix_before(1))
                with self.assertRaisesRegex(ValueError, 'extension'):
                    stream.append(history.prefix_before(1))
                from dataclasses import replace
                other = replace(history.prefix_before(1).segments[0], latent_sha256='b'*64)
                branch = history.prefix_before(0).append(other)
                with self.assertRaisesRegex(ValueError, 'extension'):
                    stream.append(branch)
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    stream.finish(history.prefix_before(1))
            finally:
                stream.abort()
            self.assertFalse(stream.thread.is_alive())

    def test_changed_latent_fails_closed_and_shutdown_joins(self):
        entered, release = threading.Event(), threading.Event()
        class BlockingDecoder(Decoder):
            def decode(self, latent):
                if self.index == 0:
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError('test release missing')
                return super().decode(latent)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(); history = Fixture(root).baseline(30)
            stream = AcceptedStream(root, history=history.prefix_before(0), output_path='test.mp4',
                chunk_count=30, decoder=BlockingDecoder(), gpu_lock=threading.RLock(),
                writer_factory=Writer, load_latent=lambda p: p.read_bytes())
            try:
                stream.append(history.prefix_before(1))
                self.assertTrue(entered.wait(10))
                stream.append(history.prefix_before(2))
                (root/history.segments[1].latent_path).write_bytes(b'tampered')
                release.set()
                self.assertTrue(stream.done.wait(10))
                with self.assertRaisesRegex(RuntimeError, 'rendering failed'):
                    stream.append(history.prefix_before(3))
            finally:
                release.set()
                stream.abort()
            self.assertFalse(stream.thread.is_alive())
            self.assertFalse((root/'test.mp4').exists())


if __name__ == '__main__':
    unittest.main()
