# The Blind Prompt Metaprompt: Design Rationale

*Companion to `metaprompt-v2.md`. This document explains the design choices behind the metaprompt and the reasoning that connects them to the intellectual claims of the course and paper.*

---

## 1. What the metaprompt must do

The blind prompt method captures intention before process begins (Day 2), then generates AI's best output from that frozen seed after process ends (Day 5). The metaprompt is what bridges the student's raw prompt — which may be vague, detailed, well-structured, or a mess — to the AI's output. It standardizes the AI's *effort* so the Friday comparison measures what human process added, not what prompting skill subtracted.

The metaprompt therefore has a dual function:

1. **Level the playing field** across students with different prompting abilities.
2. **Define what the AI brings to the comparison** — and by extension, what the comparison *demonstrates*.

The second function is the design problem. Every choice about what the AI is allowed to do determines what the gap between the AI essay and the student essay can be attributed to. Get this wrong and the demonstration is muddled.

## 2. The core tension: optimization vs. personalization

There are two competing strategies for making the AI's output as strong as possible, and they test different claims.

### Strategy A: Strongest foil

Maximize the AI's capability. Let it bring maximum knowledge, structure, argumentation. The AI produces the best essay it can from the seed, unconstrained except by the student's topic and direction.

**What this demonstrates**: Even when the AI operates at full power, the output resolves toward the central tendency of the distribution — competent, well-structured, and generic. The student's essay, by contrast, reflects committed positions that emerged through process. The gap is attributable to *commitment*, not capability.

**What this risks**: If the AI essay is clearly inferior on dimensions like factual accuracy (hallucinated citations) or domain knowledge, the comparison becomes partly about capability limits rather than purely about what process adds. The student might conclude "my essay is better because I found real sources," which is true but misses the deeper point about perspective and commitment.

**Best for**: Demonstrating that optimization cannot replace process. That the "preset" — the most probable good essay on any topic — is not the same as a *committed* essay.

### Strategy B: Steelman steerability

Try to make the AI maximally personal. Give it everything possible to produce not just a good essay but *the student's* essay — as if the AI could simulate what the student would have written if they'd had five days of process.

In practice, this would mean: allow web search (so the AI finds real sources), perhaps even provide the student's voice samples or stylistic preferences, and instruct the model to write as the student would write.

**What this demonstrates**: Even when maximally steered toward a specific person's perspective, the AI output lacks something. The "something" is the developmental trajectory — the way the student's thesis changed through reading, peer challenge, and iterative writing. The AI can simulate a destination but not the journey that produced it, and the destination itself is different because the journey shaped it.

**What this risks**: The AI might produce something that *looks* committed — that has a clear thesis, a personal voice, specific evidence. The gap might be subtler and harder for students to articulate. This makes for a more sophisticated discussion but a less dramatic demonstration.

**Best for**: Testing the harder claim — that commitment is not just a property of the output but of the process that produced it.

### The tension, stated directly

Strategy A asks: **Can optimization replace process?**
Strategy B asks: **Can steering replace commitment?**

These are different questions. The paper's framework (narrowing vs. commitment) predicts "no" to both, but for different reasons. Strategy A shows that narrowing without human steering resolves to the default. Strategy B shows that even narrowing *with* steering from a frozen seed cannot recover what iterative commitment produces, because the student's final position was not latent in the seed — it was *constructed* through the process.

## 3. The web search question

Whether to allow the AI web search access is the sharpest version of this tension.

### Without web search (recommended default)

The AI generates from training knowledge only. Its citations may be inaccurate or fabricated. This produces two visible gaps:

1. **The commitment gap**: The AI's thesis is more generic, its position less distinctive.
2. **The information gap**: The AI's sources are approximate; the student's are verified and contextualized.

Both gaps are real and both are pedagogically useful. The hallucination gap is especially vivid — students can point to specific fabricated citations and say "I found the real paper, and it actually says something different from what the AI assumed." This concretely demonstrates what research process adds.

But conflating these two gaps means students might attribute the difference primarily to information access ("the AI just didn't have the right papers") rather than commitment ("the AI didn't develop a position the way I did"). The paper's argument is about commitment, not information.

The paper (v17, Section 3) addresses this directly: "The method is parameterizable: future iterations can provide AI with information retrieval, isolating commitment from information access." The current design deliberately leaves both gaps in because (a) it's simpler, (b) both gaps are pedagogically valuable, and (c) the 2026 course is the first iteration.

### With web search

The AI gets real sources. The information gap largely disappears. What remains is the commitment gap — and if it's still visible, the demonstration is purely about what human process adds beyond what knowledge plus optimization can produce.

This is the cleaner scientific comparison. It isolates the variable the paper cares about. But it has practical costs:

- The AI essay is harder to distinguish from the student's on surface quality, making the discussion more demanding.
- The "aha moment" of seeing hallucinated citations — which is visceral and immediate — disappears.
- Search results vary by model and over time, introducing noise.

### Recommendation

**Run without web search as the primary comparison.** This is the default metaprompt configuration. The combined gap (commitment + information) is the right pedagogical demonstration for a first iteration — it's vivid, concrete, and students can point to specific differences.

**Optionally run WITH web search as a secondary comparison.** If time permits, run the same prompt with search enabled. Now students see two AI essays: one with approximate knowledge, one with real sources. The gap between *these two* shows what information access adds. The gap between the search-enabled version and the student's own essay isolates what the paper calls commitment — perspective, judgment, and the developmental trajectory of a position that changed through process.

This two-version approach gives the Friday discussion a richer structure:

- AI (no search) vs. student essay → What did your process add? (Everything: knowledge, structure, commitment)
- AI (no search) vs. AI (with search) → What does information access add? (Better sources, fewer hallucinations, possibly better evidence)
- AI (with search) vs. student essay → What does commitment add? (The hard question. The paper's question.)

## 4. Why we bind direction but free execution

The metaprompt instructs the AI to pursue the student's topic and angle but to bring maximum capability to the execution. This is not an arbitrary split. It maps onto the paper's distinction between *narrowing* (which AI can do) and *commitment* (which it cannot).

The student's Day 2 prompt provides a direction — a topic, an angle, sometimes a provisional thesis. This direction is pre-commitment: the student hasn't yet done the reading, had the peer discussions, or written through the argument that will transform their position. By binding the AI to this direction, we ensure both the AI essay and the student essay *start* from the same seed. The divergence is then attributable to what happened during the week.

If we freed the AI to choose a different direction ("develop the strongest thesis you can, even if it's not what the student described"), the comparison becomes apples-to-oranges. The AI might write a perfectly good essay on a different question, and the student can't tell whether the gap is about their process or about the AI just picking a different topic.

If we bound the AI to the student's specific claims ("only make arguments the student explicitly mentions"), we'd be handicapping the AI in a way that makes the comparison uninformative. The AI would be worse because we told it to be worse, not because it lacks the capacity for commitment.

The right split: the student's *direction* constrains the essay's subject. The AI's *capability* fills in everything else — evidence, structure, counterarguments, prose. What the comparison reveals is what the student's *process* added beyond what direction + optimization can produce.

In the paper's terms: the seed narrows the AI's distribution toward a region of topic-space. Within that region, the AI resolves to the most probable good essay — the default, the preset, the central tendency. The student, starting from the same region, navigates toward a specific committed position through a trajectory of decisions that no amount of optimization from the seed would recover. The metaprompt is designed so that this difference is what shows up on Friday.

## 5. What the comparison can and cannot show

### It can show:

- **Divergence from default resolution.** The student's essay occupies a different position in the space of possible essays on that topic than the AI's most probable version.
- **Thesis evolution.** The student's final thesis is typically absent from their Day 2 prompt. It was constructed through process. The AI cannot recover a thesis that didn't exist yet.
- **Verified vs. approximate knowledge.** The student's sources are real, contextualized, and chosen for a reason. The AI's (without search) are approximate and sometimes fabricated.
- **Individual voice.** Across seven students, seven different essays. Across seven AI outputs from seven different seeds, more homogeneity — all competent, all assistant-mode, all sounding more alike than the student essays sound.

### It cannot show:

- **That human essays are "better."** Some AI outputs may be smoother, more comprehensive, or better structured. The comparison is not about quality ranking — it's about whether the essay reflects a committed position or a default resolution.
- **That AI cannot contribute to intellectual work.** The student used AI throughout the week. The comparison is between *AI alone from a frozen seed* and *human + AI through a structured process*. It's a comparison of workflows, not of species.
- **Causation from the course structure.** We can't isolate which element of the process (solo writing, peer discussion, AI critique, reading) produced the commitment. The course structure is a package. Future iterations could ablate individual components.

## 6. The recursion

This document was written with AI. The metaprompt was designed with AI. The method it describes uses AI to generate a comparison against AI. The course it serves teaches students to think with AI by having them discover what AI cannot replace in their thinking.

This is noted not as irony but as an instance of the condition the paper describes. At each level, a human decided what the design should accomplish, what tradeoffs to make, and what claims the instrument should be able to support. The AI extended the search space (alternative framings, considerations we hadn't surfaced) and the synthesis (organizing the rationale into a coherent document). Whether this constitutes committed intellectual work or sophisticated default resolution is, appropriately, a question the reader can evaluate for themselves.

The instrument is open. So is the instrument that measures it.
