# Flux 2 Multi-Angle LoRA Support Design

## Goal

Extend this ComfyUI custom node so the same 3D camera viewer can emit prompts
for both:

- Qwen Image Edit 2511 Multiple Angles LoRA
- Flux 2 Multi-Angles LoRA v2

The node should remain a prompt/control helper. It should not load models,
apply LoRAs, run samplers, or generate images directly. ComfyUI workflows should
continue to use normal model, LoRA, text-encode, sampler, and save nodes.

## Current Behavior

The current node provides:

- a Three.js camera widget
- horizontal angle, vertical angle, and zoom controls
- optional image preview in the 3D viewer
- a prompt output
- a `camera_info` output

Current output format:

```text
<sks> {azimuth} {elevation} {distance}
```

Example:

```text
<sks> front view eye-level shot medium shot
```

This is aligned with the Qwen Image Edit multi-angle LoRA language.

## Flux 2 Target

Target LoRA:

```text
lovis93/Flux-2-Multi-Angles-LoRA-v2
```

Model card:

```text
https://huggingface.co/lovis93/Flux-2-Multi-Angles-LoRA-v2
```

The model card describes:

```text
Base model: FLUX.2-dev
ComfyUI LoRA file: flux-multi-angles-v2-72poses-comfy.safetensors
LoRA strength: 0.8 - 1.0
Prompt format: <sks> [view] [elevation] shot [distance]
72 positions: 8 azimuths x 9 elevations x 3 distances
```

The complete prompt for image generation should generally be:

```text
<sks> {view} {elevation} {distance}, {subject_prompt}
```

Examples:

```text
<sks> front view eye-level shot close-up, photo of TOK character
<sks> right side view high-angle shot medium shot, photo of TOK character
<sks> back view overhead shot wide shot, photo of TOK character
```

Note: the elevation labels already include `shot`, so do not append another
`shot` if using the literal labels below.

## Flux 2 Prompt Vocabulary

### Azimuth

Use 8 horizontal buckets.

| Angle | Prompt text |
|---:|---|
| 0 | `front view` |
| 45 | `front-right quarter view` |
| 90 | `right side view` |
| 135 | `back-right quarter view` |
| 180 | `back view` |
| 225 | `back-left quarter view` |
| 270 | `left side view` |
| 315 | `front-left quarter view` |

Mapping policy:

```text
[337.5, 360] or [0, 22.5)   -> front view
[22.5, 67.5)                -> front-right quarter view
[67.5, 112.5)               -> right side view
[112.5, 157.5)              -> back-right quarter view
[157.5, 202.5)              -> back view
[202.5, 247.5)              -> back-left quarter view
[247.5, 292.5)              -> left side view
[292.5, 337.5)              -> front-left quarter view
```

### Elevation

Flux 2 uses 9 elevation buckets, more than the current Qwen node.

| Nominal angle | Prompt text |
|---:|---|
| 0 | `eye-level shot` |
| 10 | `low-angle shot` |
| 20 | `mid-low shot` |
| 30 | `mid-angle shot` |
| 40 | `high-mid shot` |
| 45 | `high-angle shot` |
| 50 | `steep-mid shot` |
| 60 | `steep-angle shot` |
| 75 | `overhead shot` |

Suggested nearest-neighbor mapping:

```text
angle <= 5      -> eye-level shot
angle <= 15     -> low-angle shot
angle <= 25     -> mid-low shot
angle <= 35     -> mid-angle shot
angle <= 42.5   -> high-mid shot
angle <= 47.5   -> high-angle shot
angle <= 55     -> steep-mid shot
angle <= 67.5   -> steep-angle shot
else            -> overhead shot
```

Open question: the existing UI allows vertical angle from `-30` to `60`.
Flux v2's published elevation set is `0` to `75`. For implementation, prefer
one of these:

1. Add a `model_language` mode and change the vertical control range to `0..75`
   in Flux mode.
2. Keep `-30..60` for compatibility and map negative values to `low-angle shot`.

Option 1 is semantically cleaner for Flux v2. Option 2 is safer for preserving
existing UI behavior.

### Distance

Use 3 distance buckets.

| Zoom meaning | Prompt text |
|---|---|
| close | `close-up` |
| medium | `medium shot` |
| wide | `wide shot` |

The current node maps:

```text
zoom < 2     -> wide shot
zoom < 6     -> medium shot
else         -> close-up
```

This can stay as-is unless the frontend labels should be inverted to feel more
like camera radius instead of zoom.

## Proposed Node API

Keep the current node and add mode-aware outputs.

### Inputs

Existing:

```text
horizontal_angle: int
vertical_angle: int
zoom: float
default_prompts: bool
camera_view: bool
image: optional IMAGE
```

Add:

```text
model_language: enum
  - qwen_2511
  - flux2_multi_angles_v2

subject_prompt: string, optional multiline
  default: ""

include_subject: bool
  default: true

trigger_token: string
  default: "<sks>"
```

If keeping the first implementation small, `subject_prompt` and
`include_subject` can wait. The most important change is `model_language`.

### Outputs

Current:

```text
prompt: STRING
camera_info: Load3DCamera
```

Proposed:

```text
angle_prompt: STRING
full_prompt: STRING
camera_info: Load3DCamera
```

Backward-compatible option:

```text
prompt: STRING       # keep existing name, contains full prompt when subject is set
angle_prompt: STRING # new
camera_info: Load3DCamera
```

Backward compatibility matters because existing workflows may already connect
the `prompt` output. The safest first implementation is:

- keep `prompt`
- add `angle_prompt`
- keep `camera_info`

## Prompt Construction

### Qwen Mode

```python
angle_prompt = f"{trigger_token} {azimuth} {elevation} {distance}"
```

Example:

```text
<sks> front view eye-level shot medium shot
```

### Flux 2 Mode

```python
angle_prompt = f"{trigger_token} {view} {elevation} {distance}"
if subject_prompt.strip():
    full_prompt = f"{angle_prompt}, {subject_prompt.strip()}"
else:
    full_prompt = angle_prompt
```

Example:

```text
<sks> front-right quarter view mid-angle shot close-up, photo of TOK character
```

## ComfyUI Workflow Usage

The custom node should only produce prompt strings.

Flux workflow:

```text
Flux Multiangle Camera prompt/full_prompt
  -> CLIPTextEncodeFlux text input

Load Diffusion Model
  -> Load LoRA
       lora_name: flux-multi-angles-v2-72poses-comfy.safetensors
       strength_model: 0.8 - 1.0
  -> Flux sampler / guider
```

The LoRA weight belongs on the ComfyUI `Load LoRA` node, not inside this camera
node and not inside the text prompt.

## Implementation Plan

### Phase 1: Backend prompt translator

Files:

```text
nodes.py
```

Tasks:

1. Add a `model_language` combo input.
2. Add a `subject_prompt` string input.
3. Extract current mapping logic into helper functions:

   ```python
   map_azimuth(horizontal_angle) -> str
   map_qwen_elevation(vertical_angle) -> str
   map_flux2_elevation(vertical_angle) -> str
   map_distance(zoom) -> str
   build_prompt(...)
   ```

4. Preserve the current prompt output for `qwen_2511`.
5. Add Flux 2 prompt output using the vocabulary above.
6. Add focused unit-ish tests if a lightweight test harness is added later.

### Phase 2: Frontend mode selector

Files:

```text
src/App.vue
src/components/ControlPanel.vue
src/types.ts
src/i18n.ts
js/main.js
```

Tasks:

1. Add a mode selector:

   ```text
   Qwen 2511
   Flux 2 Multi-Angles v2
   ```

2. In Flux mode, expose 9 elevation presets.
3. Keep existing Qwen presets unchanged.
4. Update displayed prompt preview to match the selected mode.
5. Rebuild committed frontend output:

   ```bash
   npm install
   npm run build
   ```

### Phase 3: README and examples

Files:

```text
README.md
workflow/
```

Tasks:

1. Document Flux 2 mode.
2. Add a minimal Flux 2 workflow example.
3. Link to the LoRA model card.
4. State that users must download the ComfyUI LoRA file:

   ```text
   flux-multi-angles-v2-72poses-comfy.safetensors
   ```

5. State recommended LoRA strength:

   ```text
   0.8 - 1.0
   ```

## Compatibility Notes

- Existing Qwen workflows should continue to load.
- The existing `default_prompts` input is deprecated but should remain until a
  larger cleanup.
- The node should not hardcode Flux model paths or LoRA paths.
- Do not make this node responsible for applying LoRA strength.
- Keep the prompt output English-only, matching current behavior.

## Risks

- The Flux LoRA expects a precise prompt dialect. Small vocabulary changes may
  reduce control quality.
- The current vertical angle UI range does not exactly match Flux v2's 9
  elevation set.
- Existing workflows may break if output names or input ordering change too
  aggressively.
- Frontend state persistence may need a version bump if new mode-specific state
  is stored.

## Acceptance Criteria

1. Existing Qwen prompt output remains unchanged in Qwen mode.
2. Flux mode can emit all 72 Flux v2 camera prompt combinations.
3. Flux mode examples match the model card prompt format.
4. A user can connect `full_prompt` or `prompt` to `CLIPTextEncodeFlux`.
5. The LoRA remains loaded through normal ComfyUI `Load LoRA` nodes.
6. The built `js/` bundle is committed so normal users do not need Node.js.

