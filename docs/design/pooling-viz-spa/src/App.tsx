import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Html, OrbitControls, RoundedBox, Text } from '@react-three/drei';
import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';
import * as THREE from 'three';

type Mode =
  | 'Overview'
  | 'Pooling Pipeline'
  | 'Pooling Types'
  | 'vLLM vs Generation'
  | 'vLLM vs Naive Embedding Server'
  | 'Benchmark Simulator';

type PoolType = 'CLS' | 'LAST' | 'MEAN' | 'ALL' | 'STEP';
type TaskType = 'embed' | 'classify' | 'token_embed' | 'token_classify';

const modes: Mode[] = ['Overview', 'Pooling Pipeline', 'Pooling Types', 'vLLM vs Generation', 'vLLM vs Naive Embedding Server', 'Benchmark Simulator'];

function CubeToken({ pos, color = '#3b82f6' }: { pos: [number, number, number]; color?: string }) {
  const ref = useMemo(() => new THREE.Object3D(), []);
  useFrame(({ clock }) => {
    ref.position.set(pos[0], pos[1] + Math.sin(clock.elapsedTime + pos[0]) * 0.04, pos[2]);
  });
  return (
    <group ref={(g) => g && g.add(ref)}>
      <mesh position={pos}>
        <boxGeometry args={[0.18, 0.18, 0.18]} />
        <meshStandardMaterial color={color} />
      </mesh>
    </group>
  );
}

function TensorBlock({ size = [2.6, 1.2, 1], color = '#22c55e' }: { size?: [number, number, number]; color?: string }) {
  return (
    <mesh position={[0.8, 0, 0]}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} opacity={0.32} transparent />
    </mesh>
  );
}

function PoolerMachine() {
  return (
    <group position={[3.3, 0, 0]}>
      <RoundedBox args={[1.2, 1.4, 1.2]} radius={0.08}>
        <meshStandardMaterial color="#f59e0b" />
      </RoundedBox>
      <Text position={[0, 1.0, 0]} fontSize={0.2} color="black">POOLER</Text>
    </group>
  );
}

function Scene({ mode, poolType, taskType, chapter, util, overhead }: { mode: Mode; poolType: PoolType; taskType: TaskType; chapter: number; util: number; overhead: number }) {
  const tokens = Array.from({ length: 24 }, (_, i) => [i % 8, Math.floor(i / 8), 0] as [number, number, number]);
  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 7, 5]} intensity={1.3} />
      <OrbitControls enablePan enableZoom />

      {(mode === 'Overview' || mode === 'Pooling Pipeline' || mode === 'Pooling Types') && (
        <group position={[-4, -1.2, 0]}>
          <Text position={[-0.5, 2.3, 0]} fontSize={0.22} color="black">Input Queue</Text>
          {tokens.map((p, idx) => <CubeToken key={idx} pos={[p[0] * 0.24, p[1] * 0.24, 0]} />)}
          <Float speed={1.5}>
            <TensorBlock />
          </Float>
          <Text position={[0.8, 1.2, 0.7]} fontSize={0.17} color="black">hidden states [B,S,H]</Text>
          <PoolerMachine />
          <mesh position={[5.4, 0, 0]}>
            <sphereGeometry args={[0.4, 32, 32]} />
            <meshStandardMaterial color="#8b5cf6" />
          </mesh>
          <Text position={[5.4, 0.8, 0]} fontSize={0.16} color="black">output</Text>
          <Text position={[3.2, -1.25, 0]} fontSize={0.14} color="black">pooling={poolType}</Text>

          {mode === 'Pooling Types' && (
            <group>
              {poolType === 'CLS' && <mesh position={[0, 0.5, 0.75]}><boxGeometry args={[0.2, 0.2, 0.2]} /><meshStandardMaterial color="#fde047" /></mesh>}
              {poolType === 'LAST' && <mesh position={[1.68, 0.02, 0.75]}><boxGeometry args={[0.2, 0.2, 0.2]} /><meshStandardMaterial color="#fde047" /></mesh>}
              {poolType === 'MEAN' && <Text position={[0.8, -0.7, 0.8]} fontSize={0.14} color="black">average over valid tokens</Text>}
              {poolType === 'ALL' && <Text position={[5.4, -0.7, 0.8]} fontSize={0.14} color="black">return all token states</Text>}
              {poolType === 'STEP' && <Text position={[2.2, -0.7, 0.8]} fontSize={0.14} color="black">filter by returned token IDs</Text>}
            </group>
          )}

          {mode === 'Pooling Pipeline' && (
            <group position={[6.7, 0, 0]}>
              <Text position={[0, 1.2, 0]} fontSize={0.15} color="black">task={taskType}</Text>
              {taskType === 'embed' && <mesh><octahedronGeometry args={[0.45]} /><meshStandardMaterial color="#8b5cf6" /></mesh>}
              {taskType === 'classify' && (
                <group>
                  {[0.3, 0.7, 1.0].map((h, i) => <mesh key={i} position={[i * 0.4 - 0.4, h / 2 - 0.4, 0]}><boxGeometry args={[0.2, h, 0.2]} /><meshStandardMaterial color="#a78bfa" /></mesh>)}
                </group>
              )}
              {(taskType === 'token_embed' || taskType === 'token_classify') && (
                <group>{Array.from({ length: 6 }, (_, i) => <mesh key={i} position={[i * 0.28 - 0.7, 0, 0]}><sphereGeometry args={[0.11]} /><meshStandardMaterial color="#8b5cf6" /></mesh>)}</group>
              )}
            </group>
          )}
        </group>
      )}

      {mode === 'vLLM vs Generation' && (
        <group position={[-5.4, -1.2, 0]}>
          <Text position={[1.7, 2.5, 0]} fontSize={0.2} color="black">Generation lane</Text>
          <Text position={[7.1, 2.5, 0]} fontSize={0.2} color="black">Pooling lane</Text>
          {['prefill', 'kv cache', 'decode', 'sample', 'append', 'decode', 'sample'].map((s, i) => (
            <mesh key={s + i} position={[i * 1.1, 1.6, 0]}>
              <boxGeometry args={[0.95, 0.35, 0.25]} /><meshStandardMaterial color={i % 2 ? '#f87171' : '#60a5fa'} />
              <Html center><div className="text-[10px] bg-black text-white px-1 rounded">{s}</div></Html>
            </mesh>
          ))}
          {['input', 'forward', 'pooler', 'output'].map((s, i) => (
            <mesh key={s} position={[6 + i * 1.2, 0.5, 0]}>
              <boxGeometry args={[1, 0.35, 0.25]} /><meshStandardMaterial color="#34d399" />
              <Html center><div className="text-[10px] bg-black text-white px-1 rounded">{s}</div></Html>
            </mesh>
          ))}
        </group>
      )}

      {mode === 'vLLM vs Naive Embedding Server' && (
        <group position={[-4.5, -1.2, 0]}>
          <Text position={[1.6, 2.4, 0]} fontSize={0.2} color="black">Naive server</Text>
          <Text position={[6.8, 2.4, 0]} fontSize={0.2} color="black">vLLM-style serving</Text>
          {[0.5, 0.3, 0.7, 0.2].map((h, i) => <mesh key={i} position={[i * 0.9, h / 2 + 0.6, 0]}><boxGeometry args={[0.6, h, 0.3]} /><meshStandardMaterial color="#94a3b8" /></mesh>)}
          <mesh position={[2.1, 0.9, 0]}><boxGeometry args={[0.8, overhead / 100 + 0.3, 0.4]} /><meshStandardMaterial color="#ef4444" /></mesh>
          {[0.7, 0.9, 1.2, 1.1].map((h, i) => <mesh key={i} position={[5.4 + i * 0.9, h / 2 + 0.5, 0]}><boxGeometry args={[0.7, h, 0.3]} /><meshStandardMaterial color="#22c55e" /></mesh>)}
          <mesh position={[8.8, util / 200, 0]}><boxGeometry args={[0.7, util / 100 + 0.4, 0.35]} /><meshStandardMaterial color="#3b82f6" /></mesh>
        </group>
      )}

      {mode === 'Benchmark Simulator' && (
        <group position={[-2, -0.4, 0]}>
          <Text position={[0, 2.4, 0]} fontSize={0.22} color="black">Conceptual Simulator (Toy Model)</Text>
          <mesh position={[0, 1.2, 0]}><boxGeometry args={[4, 0.2, 0.2]} /><meshStandardMaterial color="#cbd5e1" /></mesh>
          <mesh position={[-1.8 + chapter * 0.5, 1.2, 0.1]}><sphereGeometry args={[0.12]} /><meshStandardMaterial color="#2563eb" /></mesh>
        </group>
      )}
    </>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>('Overview');
  const [chapter, setChapter] = useState(1);
  const [poolType, setPoolType] = useState<PoolType>('CLS');
  const [taskType, setTaskType] = useState<TaskType>('embed');
  const [batchSize, setBatchSize] = useState(32);
  const [seqLen, setSeqLen] = useState(256);
  const [hiddenDim, setHiddenDim] = useState(1024);
  const [numReq, setNumReq] = useState(400);
  const [overhead, setOverhead] = useState(30);
  const [util, setUtil] = useState(70);

  const estimates = useMemo(() => {
    const baseForward = (batchSize * seqLen * hiddenDim) / 3e7;
    const naiveLatency = baseForward + overhead / 20 + numReq / 350;
    const poolingLatency = baseForward * 0.72 + (100 - util) / 60 + numReq / 420;
    const throughput = Math.max(1, (batchSize * 1000) / (poolingLatency * 10));
    return { naiveLatency, poolingLatency, throughput };
  }, [batchSize, seqLen, hiddenDim, overhead, util, numReq]);

  const chapterText = {
    1: 'vLLM pooling model은 VllmModelForPooling + Pooler 구조를 통해 final hidden states에서 task별 출력을 만든다.',
    2: 'Pooling task: embed / classify / token_embed / token_classify.',
    3: 'Pooling type: CLS / LAST / MEAN / ALL / STEP.',
    4: 'Generation path는 token-by-token decode loop가 필요하지만 pooling은 one-pass forward + pooling으로 끝난다.',
    5: '성능은 모델, batch size, seq len, backend, vLLM 버전에 따라 달라진다.'
  } as const;

  return (
    <div className="h-full w-full bg-slate-100 text-slate-900">
      <header className="border-b bg-white px-6 py-4">
        <h1 className="text-2xl font-semibold">vLLM Pooling Task Visualization</h1>
        <div className="mt-3 flex flex-wrap gap-2">
          {modes.map((m) => (
            <button key={m} onClick={() => setMode(m)} className={`rounded-full px-3 py-1 text-sm ${mode === m ? 'bg-blue-600 text-white' : 'bg-slate-200'}`}>{m}</button>
          ))}
        </div>
      </header>

      <div className="grid h-[calc(100%-98px)] grid-cols-1 lg:grid-cols-[380px_1fr]">
        <motion.aside initial={{ x: -15, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="overflow-auto border-r bg-white p-4">
          <h2 className="text-lg font-semibold">Walkthrough</h2>
          <p className="mt-2 text-sm">Chapter {chapter}/5</p>
          <input type="range" min={1} max={5} value={chapter} onChange={(e) => setChapter(Number(e.target.value))} className="mt-2 w-full" />
          <p className="mt-2 rounded bg-slate-100 p-3 text-sm">{chapterText[chapter as 1 | 2 | 3 | 4 | 5]}</p>
          <div className="mt-2 flex gap-2">
            <button className="rounded bg-slate-200 px-3 py-1" onClick={() => setChapter((c) => Math.max(1, c - 1))}>Prev</button>
            <button className="rounded bg-slate-200 px-3 py-1" onClick={() => setChapter((c) => Math.min(5, c + 1))}>Next</button>
          </div>

          <h3 className="mt-4 font-semibold">Controls</h3>
          <label className="mt-2 block text-sm">Pooling type</label>
          <select className="w-full rounded border p-1" value={poolType} onChange={(e) => setPoolType(e.target.value as PoolType)}>
            {['CLS', 'LAST', 'MEAN', 'ALL', 'STEP'].map((x) => <option key={x}>{x}</option>)}
          </select>

          <label className="mt-2 block text-sm">Task type</label>
          <select className="w-full rounded border p-1" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
            {['embed', 'classify', 'token_embed', 'token_classify'].map((x) => <option key={x}>{x}</option>)}
          </select>

          <div className="mt-4 rounded border border-amber-300 bg-amber-50 p-3 text-sm">
            <b>Caveat</b>: PagedAttention/KV cache 최적화는 autoregressive generation에서 효과가 크며, pooling은 decode loop가 없어 의미가 다를 수 있습니다.
          </div>

          <pre className="mt-4 overflow-auto rounded bg-slate-900 p-3 text-xs text-slate-100">{`# vLLM pooling conceptual flow
hidden_states = model.forward(input_ids)
pooled = pooler(hidden_states, metadata)
# embed/classify/token outputs
return pooled`}</pre>

          {mode === 'Benchmark Simulator' && (
            <div className="mt-4 space-y-2 rounded border p-3 text-sm">
              <p className="font-semibold">Toy Benchmark Inputs</p>
              {[
                ['batch size', batchSize, setBatchSize, 1, 128],
                ['avg sequence length', seqLen, setSeqLen, 16, 2048],
                ['hidden dimension', hiddenDim, setHiddenDim, 128, 4096],
                ['number of requests', numReq, setNumReq, 10, 2000],
                ['naive server overhead', overhead, setOverhead, 0, 100],
                ['GPU utilization', util, setUtil, 10, 100],
              ].map(([name, value, setter, min, max]) => (
                <div key={name as string}>
                  <label className="text-xs">{name}: {value as number}</label>
                  <input className="w-full" type="range" min={min as number} max={max as number} value={value as number} onChange={(e) => (setter as (n:number)=>void)(Number(e.target.value))} />
                </div>
              ))}
              <p>Conceptual latency (naive): <b>{estimates.naiveLatency.toFixed(2)} ms</b></p>
              <p>Conceptual latency (vLLM-style): <b>{estimates.poolingLatency.toFixed(2)} ms</b></p>
              <p>Conceptual throughput: <b>{estimates.throughput.toFixed(1)} req/s</b></p>
              <p className="rounded bg-red-100 p-2 text-xs">Not a real benchmark. Toy model only.</p>
            </div>
          )}
        </motion.aside>

        <main className="h-full">
          <Canvas camera={{ position: [2, 3.4, 10], fov: 42 }}>
            <Scene mode={mode} poolType={poolType} taskType={taskType} chapter={chapter} util={util} overhead={overhead} />
          </Canvas>
        </main>
      </div>
    </div>
  );
}
