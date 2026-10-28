/**
 * System 1 Engine — Standalone Web Evaluation Runtime
 * Permite que qualquer portfólio importe e execute o agente neural diretamente no navegador via ONNX Runtime Web.
 * 
 * GitHub: https://github.com/JulioCRFilho/system_one
 */

// =====================================================================
// SIMULADOR DO CUBO MÁGICO 3X3 EM JAVASCRIPT PURO
// =====================================================================

class RubiksCubeSim {
  static ATOMIC_MOVES = [
    "U", "U_prime", "D", "D_prime",
    "F", "F_prime", "B", "B_prime",
    "R", "R_prime", "L", "L_prime"
  ];

  static MACRO_ACTIONS = {
    "SEXY_MOVE_R": ["R", "U", "R_prime", "U_prime"],
    "SEXY_MOVE_R_PRIME": ["U", "R", "U_prime", "R_prime"],
    "SEXY_MOVE_L": ["L_prime", "U_prime", "L", "U"],
    "SEXY_MOVE_L_PRIME": ["U_prime", "L_prime", "U", "L"],
    "SUNE": ["R", "U", "R_prime", "U", "R", "U", "U", "R_prime"],
    "ANTI_SUNE": ["R", "U", "U", "R_prime", "U_prime", "R", "U_prime", "R_prime"],
    "YELLOW_CROSS": ["F", "R", "U", "R_prime", "U_prime", "F_prime"],
    "YELLOW_CROSS_PRIME": ["F", "U", "R", "U_prime", "R_prime", "F_prime"],
    "ROTATE_Y": ["Y"],
    "ROTATE_Y_PRIME": ["Y_prime"],
    "U_TURN": ["U"],
    "U_PRIME_TURN": ["U_prime"]
  };
  static MACRO_NAMES = Object.keys(RubiksCubeSim.MACRO_ACTIONS);

  static INVERSE_ATOMIC = {
    "U": "U_prime", "U_prime": "U",
    "D": "D_prime", "D_prime": "D",
    "F": "F_prime", "F_prime": "F",
    "B": "B_prime", "B_prime": "B",
    "R": "R_prime", "R_prime": "R",
    "L": "L_prime", "L_prime": "L"
  };

  static INVERSE_MACROS = {
    "SEXY_MOVE_R": "SEXY_MOVE_R_PRIME",
    "SEXY_MOVE_R_PRIME": "SEXY_MOVE_R",
    "SEXY_MOVE_L": "SEXY_MOVE_L_PRIME",
    "SEXY_MOVE_L_PRIME": "SEXY_MOVE_L",
    "SUNE": "ANTI_SUNE",
    "ANTI_SUNE": "SUNE",
    "YELLOW_CROSS": "YELLOW_CROSS_PRIME",
    "YELLOW_CROSS_PRIME": "YELLOW_CROSS",
    "ROTATE_Y": "ROTATE_Y_PRIME",
    "ROTATE_Y_PRIME": "ROTATE_Y",
    "U_TURN": "U_PRIME_TURN",
    "U_PRIME_TURN": "U_TURN"
  };

  static SCRAMBLE_MACRO_NAMES = [
    "SEXY_MOVE_R", "SEXY_MOVE_R_PRIME",
    "SEXY_MOVE_L", "SEXY_MOVE_L_PRIME",
    "SUNE", "ANTI_SUNE",
    "YELLOW_CROSS", "YELLOW_CROSS_PRIME",
    "U_TURN", "U_PRIME_TURN"
  ];

  static PALETTE = {
    0: "#F5F5FA", // U: Branco
    1: "#FACC15", // D: Amarelo
    2: "#22C55E", // F: Verde
    3: "#3B82F6", // B: Azul
    4: "#EF4444", // R: Vermelho
    5: "#F97316"  // L: Laranja
  };

  static FACE_LAYOUT = {
    0: { x: 164, y: 46 },  // U
    5: { x: 88,  y: 122 }, // L
    2: { x: 164, y: 122 }, // F
    4: { x: 240, y: 122 }, // R
    3: { x: 316, y: 122 }, // B
    1: { x: 164, y: 198 }  // D
  };

  constructor() {
    this.permutations = this._initPermutations();
    this.scrambleSequence = [];
    this._initWasm();
    this.reset();
  }

  _initWasm() {
    if (typeof window !== "undefined" && window.System1Core && window.System1Core.RubiksCubeCoreWasm) {
      try {
        this._wasm = new window.System1Core.RubiksCubeCoreWasm();
      } catch (_) {
        this._wasm = null;
      }
    } else {
      this._wasm = null;
    }
  }

  _initPermutations() {
    // Rotações 3D matemáticas idênticas ao RubiksCubeCore
    const rotF = [ [0, 1, 0], [-1, 0, 0], [0, 0, 1] ];
    const rotB = [ [0, -1, 0], [1, 0, 0], [0, 0, 1] ];
    const rotR = [ [1, 0, 0], [0, 0, -1], [0, 1, 0] ];
    const rotL = [ [1, 0, 0], [0, 0, 1], [0, -1, 0] ];
    const rotU = [ [0, 0, -1], [0, 1, 0], [1, 0, 0] ];
    const rotD = [ [0, 0, 1], [0, 1, 0], [-1, 0, 0] ];

    const transpose = (m) => [
      [m[0][0], m[1][0], m[2][0]],
      [m[0][1], m[1][1], m[2][1]],
      [m[0][2], m[1][2], m[2][2]]
    ];

    const moves = {
      "F": { m: rotF, cond: (p) => p[2] === 1 },
      "F_prime": { m: transpose(rotF), cond: (p) => p[2] === 1 },
      "B": { m: rotB, cond: (p) => p[2] === -1 },
      "B_prime": { m: transpose(rotB), cond: (p) => p[2] === -1 },
      "R": { m: rotR, cond: (p) => p[0] === 1 },
      "R_prime": { m: transpose(rotR), cond: (p) => p[0] === 1 },
      "L": { m: rotL, cond: (p) => p[0] === -1 },
      "L_prime": { m: transpose(rotL), cond: (p) => p[0] === -1 },
      "U": { m: rotU, cond: (p) => p[1] === 1 },
      "U_prime": { m: transpose(rotU), cond: (p) => p[1] === 1 },
      "D": { m: rotD, cond: (p) => p[1] === -1 },
      "D_prime": { m: transpose(rotD), cond: (p) => p[1] === -1 },
      "Y": { m: rotU, cond: () => true },
      "Y_prime": { m: transpose(rotU), cond: () => true }
    };

    const getSlot = (pos, norm) => {
      const [x, y, z] = pos;
      const [nx, ny, nz] = norm;
      let face, r, c;
      if (ny === 1) { face = 0; r = 1 - z; c = x + 1; }
      else if (ny === -1) { face = 1; r = z + 1; c = x + 1; }
      else if (nz === 1) { face = 2; r = 1 - y; c = x + 1; }
      else if (nz === -1) { face = 3; r = 1 - y; c = 1 - x; }
      else if (nx === 1) { face = 4; r = 1 - y; c = 1 - z; }
      else if (nx === -1) { face = 5; r = 1 - y; c = z + 1; }
      return face * 9 + (r * 3 + c);
    };

    const perms = {};
    for (const [mName, { m: rotMat, cond }] of Object.entries(moves)) {
      const stickers = [];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1, 1 - r], norm: [0, 1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, -1, r - 1], norm: [0, -1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1 - r, 1], norm: [0, 0, 1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1 - c, 1 - r, -1], norm: [0, 0, -1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1, 1 - r, 1 - c], norm: [1, 0, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [-1, 1 - r, c - 1], norm: [-1, 0, 0] });

      for (let i = 0; i < stickers.length; i++) stickers[i].srcSlot = getSlot(stickers[i].pos, stickers[i].norm);

      const mul = (mat, v) => [
        mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2],
        mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2],
        mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
      ];

      const perm = new Int32Array(54);
      for (const s of stickers) {
        let p = s.pos, n = s.norm;
        if (cond(p)) { p = mul(rotMat, p); n = mul(rotMat, n); }
        const destSlot = getSlot(p, n);
        perm[destSlot] = s.srcSlot;
      }
      perms[mName] = perm;
    }
    return perms;
  }

  reset() {
    if (!this._wasm && typeof window !== "undefined" && window.System1Core && window.System1Core.RubiksCubeCoreWasm) {
      this._initWasm();
    }
    if (this._wasm) {
      this._wasm.reset();
      this.state = this._wasm.get_state();
    } else {
      this.state = new Int32Array(54);
      for (let f = 0; f < 6; f++) {
        for (let i = 0; i < 9; i++) this.state[f * 9 + i] = f;
      }
    }
    this.lastAction = "RESET";
    this.scrambleSequence = [];
    this.steps = 0;
  }

  applyAtomic(moveName, countStep = true) {
    if (this._wasm) {
      this._wasm.apply_atomic(moveName);
      this.state = this._wasm.get_state();
    } else {
      const perm = this.permutations[moveName];
      if (!perm) return;
      const nextState = new Int32Array(54);
      for (let i = 0; i < 54; i++) nextState[i] = this.state[perm[i]];
      this.state = nextState;
    }
    if (countStep) {
      this.lastAction = moveName;
      this.steps++;
    }
  }

  predictStateHash(moveName) {
    const perm = this.permutations[moveName];
    if (!perm) return "";
    const nextState = new Int32Array(54);
    for (let i = 0; i < 54; i++) nextState[i] = this.state[perm[i]];
    return nextState.join("");
  }

  getStateHash() {
    return this.state.join("");
  }

  applyMacro(macroName) {
    if (this._wasm) {
      this._wasm.apply_macro(macroName);
      this.state = this._wasm.get_state();
    } else {
      const seq = RubiksCubeSim.MACRO_ACTIONS[macroName];
      if (!seq) return;
      for (const m of seq) this.applyAtomic(m, false);
    }
    this.lastAction = macroName;
    this.steps++;
  }

  scrambleAtomic(depth = 3) {
    this.reset();
    const applied = [];
    let lastMove = null;
    let attempts = 0;
    const targetDepth = Math.max(1, parseInt(depth, 10) || 3);
    const maxAttempts = Math.max(100, targetDepth * 10);
    while ((applied.length < targetDepth || this.isSolved()) && attempts < maxAttempts) {
      attempts++;
      const candidates = RubiksCubeSim.ATOMIC_MOVES.filter(m => m !== RubiksCubeSim.INVERSE_ATOMIC[lastMove]);
      const move = candidates[Math.floor(Math.random() * candidates.length)];
      this.applyAtomic(move, false);
      applied.push(move);
      lastMove = move;
    }
    this.lastAction = "EMBARALHADO";
    this.scrambleSequence = applied;
    this.steps = 0;
    return applied;
  }

  scrambleMacro(depth = 2) {
    this.reset();
    const applied = [];
    let lastMove = null;
    let attempts = 0;
    const targetDepth = Math.max(1, parseInt(depth, 10) || 2);
    const maxAttempts = Math.max(100, targetDepth * 10);
    while ((applied.length < targetDepth || this.isSolved()) && attempts < maxAttempts) {
      attempts++;
      const candidates = RubiksCubeSim.SCRAMBLE_MACRO_NAMES.filter(m => m !== RubiksCubeSim.INVERSE_MACROS[lastMove]);
      const move = candidates[Math.floor(Math.random() * candidates.length)];
      this.applyMacro(move);
      applied.push(move);
      lastMove = move;
    }
    this.lastAction = "EMBARALHADO";
    this.scrambleSequence = applied;
    this.steps = 0;
    return applied;
  }

  scramble(depth = 3, mode = "atomic") {
    const d = Math.max(1, parseInt(depth, 10) || (mode === "macro" ? 2 : 3));
    if (this._wasm) {
      const moves = this._wasm.scramble(d, mode === "macro");
      this.state = this._wasm.get_state();
      this.lastAction = "EMBARALHADO";
      this.scrambleSequence = moves;
      this.steps = 0;
      return moves;
    }
    if (mode === "macro") {
      return this.scrambleMacro(d);
    }
    return this.scrambleAtomic(d);
  }

  getAlignedCount() {
    if (this._wasm) return this._wasm.get_aligned_count();
    let count = 0;
    for (let f = 0; f < 6; f++) {
      const center = this.state[f * 9 + 4];
      for (let i = 0; i < 9; i++) {
        if (this.state[f * 9 + i] === center) count++;
      }
    }
    return count;
  }

  getScore() {
    if (this._wasm) return this._wasm.get_score();
    const aligned = this.getAlignedCount();
    return Math.max(0.0, Math.min(1.0, (aligned - 6) / 48.0));
  }

  isSolved() {
    if (this._wasm) return this._wasm.is_solved();
    return this.getAlignedCount() === 54;
  }

  getOneHot() {
    if (this._wasm) return this._wasm.get_one_hot();
    const arr = new Float32Array(324);
    for (let i = 0; i < 54; i++) {
      const color = this.state[i];
      arr[i * 6 + color] = 1.0;
    }
    return arr;
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Fundo Cyber Dark
    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);

    // Moldura
    ctx.strokeStyle = '#1F2937';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    // Cabeçalho
    const aligned = this.getAlignedCount();
    const scorePct = ((aligned - 6) / 48 * 100).toFixed(1);
    ctx.fillStyle = '#38BDF8';
    ctx.font = 'bold 12px monospace';
    ctx.fillText("⚡ SYSTEM 1 HUD | CUBO MÁGICO 3X3", 20, 26);

    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px monospace';
    let statusLine = `Passos: ${this.steps} | Alinhamento: ${aligned}/54 (${scorePct}%)`;
    if (this.scrambleSequence && this.scrambleSequence.length > 0 && this.steps === 0) {
      statusLine += ` | Scramble: ${this.scrambleSequence.join(" ")}`;
    }
    ctx.fillText(statusLine, 20, 42);

    // Desenho das 6 faces em Cruz
    for (const [fIdxStr, { x: fx, y: fy }] of Object.entries(RubiksCubeSim.FACE_LAYOUT)) {
      const fIdx = parseInt(fIdxStr);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const stickerIdx = fIdx * 9 + (r * 3 + c);
          const colId = this.state[stickerIdx];
          ctx.fillStyle = RubiksCubeSim.PALETTE[colId] || '#FFFFFF';
          const x1 = fx + c * 24;
          const y1 = fy + r * 24;
          ctx.fillRect(x1, y1, 22, 22);
          ctx.strokeStyle = '#0F172A';
          ctx.lineWidth = 1;
          ctx.strokeRect(x1, y1, 22, 22);
        }
      }
    }

    // Rodapé
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = this.isSolved() ? '#4ADE80' : '#E2E8F0';
    ctx.fillText(this.isSolved() ? "STATUS: RESOLVIDO 🏆" : `AÇÃO ATIVA: ${this.lastAction}`, 20, height - 16);
  }
}

// =====================================================================
// SIMULADOR CARTPOLE-V1 EM JAVASCRIPT PURO
// =====================================================================

class CartPoleSim {
  constructor() {
    this.gravity = 9.8;
    this.masscart = 1.0;
    this.masspole = 0.1;
    this.total_mass = this.masscart + this.masspole;
    this.length = 0.5;
    this.polemass_length = this.masspole * this.length;
    this.force_mag = 10.0;
    this.tau = 0.02;
    this.reset();
  }

  reset() {
    this.x = (Math.random() - 0.5) * 0.1;
    this.x_dot = (Math.random() - 0.5) * 0.1;
    this.theta = (Math.random() - 0.5) * 0.1;
    this.theta_dot = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
  }

  getState() {
    return new Float32Array([this.x, this.x_dot, this.theta, this.theta_dot]);
  }

  step(action) {
    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costh = Math.cos(this.theta);
    const sinth = Math.sin(this.theta);

    const temp = (force + this.polemass_length * this.theta_dot * this.theta_dot * sinth) / this.total_mass;
    const thetaacc = (this.gravity * sinth - costh * temp) / (this.length * (4.0 / 3.0 - this.masspole * costh * costh / this.total_mass));
    const xacc = temp - this.polemass_length * thetaacc * costh / this.total_mass;

    this.x += this.tau * this.x_dot;
    this.x_dot += this.tau * xacc;
    this.theta += this.tau * this.theta_dot;
    this.theta_dot += this.tau * thetaacc;
    this.steps++;

    const done = Math.abs(this.x) > 2.4 || Math.abs(this.theta) > 0.2095;
    return { done, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#1F2937';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    ctx.fillStyle = '#38BDF8';
    ctx.font = 'bold 12px monospace';
    ctx.fillText("⚡ SYSTEM 1 HUD | CARTPOLE-V1", 20, 26);
    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px monospace';
    ctx.fillText(`Passos Equilibrados: ${this.steps}`, 20, 42);

    // Linha do solo
    const groundY = height - 80;
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, groundY);
    ctx.lineTo(width - 20, groundY);
    ctx.stroke();

    // Carrinho
    const scale = width / 4.8;
    const cartX = width / 2 + this.x * scale;
    const cartW = 60, cartH = 30;
    ctx.fillStyle = '#38BDF8';
    ctx.fillRect(cartX - cartW/2, groundY - cartH, cartW, cartH);

    // Haste
    const poleLen = 90;
    const poleEndX = cartX + Math.sin(this.theta) * poleLen;
    const poleEndY = (groundY - cartH) - Math.cos(this.theta) * poleLen;
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartX, groundY - cartH);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
  }
}

// =====================================================================
// SIMULADOR LUNARLANDER-V3 (POUSO ESPACIAL EM JS PURO)
// =====================================================================

class LunarLanderSim {
  constructor() {
    this.groundY = 0.18;
    this.x = 0;
    this.y = 1.3;
    this.vx = 0;
    this.vy = 0;
    this.angle = 0;
    this.v_angle = 0;
    this.leftContact = 0;
    this.rightContact = 0;
    this.steps = 0;
    this.fuel = 100;
    this.lastAction = 0;
    this.isLanded = false;
    this.isCrashed = false;
    this.particles = [];
    this.reset();
  }

  reset() {
    this.x = (Math.random() - 0.5) * 0.4;
    this.y = 1.3 + Math.random() * 0.1;
    this.vx = (Math.random() - 0.5) * 0.2;
    this.vy = -(0.02 + Math.random() * 0.04);
    this.angle = (Math.random() - 0.5) * 0.1;
    this.v_angle = (Math.random() - 0.5) * 0.05;
    this.leftContact = 0;
    this.rightContact = 0;
    this.steps = 0;
    this.fuel = 100;
    this.lastAction = 0;
    this.isLanded = false;
    this.isCrashed = false;
    this.particles = [];
  }

  getState() {
    return new Float32Array([
      this.x,
      this.y,
      this.vx,
      this.vy,
      this.angle,
      this.v_angle,
      this.leftContact,
      this.rightContact,
    ]);
  }

  step(action) {
    this.lastAction = action;
    this.steps++;

    const dt = 0.02;
    const gravity = -3.2;
    const mainThrust = 9.0;
    const sideThrust = 2.4;

    let thrustX = 0;
    let thrustY = gravity;
    let torque = 0;

    // Ações: 0=Noop, 1=Propulsor Esquerdo, 2=Principal, 3=Propulsor Direito
    if (action === 2 && this.fuel > 0) {
      thrustX -= Math.sin(this.angle) * mainThrust;
      thrustY += Math.cos(this.angle) * mainThrust;
      this.fuel = Math.max(0, this.fuel - 0.12);

      for (let i = 0; i < 3; i++) {
        this.particles.push({
          x: this.x - Math.sin(this.angle) * 0.08,
          y: this.y - Math.cos(this.angle) * 0.08,
          vx: -Math.sin(this.angle) * (1.2 + Math.random()) + (Math.random() - 0.5) * 0.3,
          vy: -Math.cos(this.angle) * (1.2 + Math.random()) + (Math.random() - 0.5) * 0.3,
          life: 1.0,
          color: Math.random() > 0.4 ? '#f59e0b' : '#ef4444',
        });
      }
    } else if (action === 1 && this.fuel > 0) {
      torque -= sideThrust;
      thrustX += Math.cos(this.angle) * 0.6;
      this.fuel = Math.max(0, this.fuel - 0.06);

      for (let i = 0; i < 2; i++) {
        this.particles.push({
          x: this.x - Math.cos(this.angle) * 0.06,
          y: this.y + Math.sin(this.angle) * 0.06,
          vx: -Math.cos(this.angle) * 0.7,
          vy: Math.sin(this.angle) * 0.7,
          life: 0.8,
          color: '#38bdf8',
        });
      }
    } else if (action === 3 && this.fuel > 0) {
      torque += sideThrust;
      thrustX -= Math.cos(this.angle) * 0.6;
      this.fuel = Math.max(0, this.fuel - 0.06);

      for (let i = 0; i < 2; i++) {
        this.particles.push({
          x: this.x + Math.cos(this.angle) * 0.06,
          y: this.y - Math.sin(this.angle) * 0.06,
          vx: Math.cos(this.angle) * 0.7,
          vy: -Math.sin(this.angle) * 0.7,
          life: 0.8,
          color: '#38bdf8',
        });
      }
    }

    this.vx += thrustX * dt;
    this.vy += thrustY * dt;
    this.v_angle += torque * dt;

    this.vx *= 0.998;
    this.v_angle *= 0.96;

    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.angle += this.v_angle * dt;

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= dt * 2.5;
      if (p.life <= 0) this.particles.splice(i, 1);
    }

    let done = false;

    if (this.y <= this.groundY) {
      this.y = this.groundY;
      const speed = Math.hypot(this.vx, this.vy);
      const angleDeg = Math.abs((this.angle * 180) / Math.PI);
      const onPad = Math.abs(this.x) < 0.28;

      if (speed < 0.45 && angleDeg < 15 && onPad) {
        this.isLanded = true;
        this.leftContact = 1;
        this.rightContact = 1;
        this.vx = 0;
        this.vy = 0;
        this.v_angle = 0;
        done = true;
      } else {
        this.isCrashed = true;
        done = true;
      }
    }

    if (Math.abs(this.x) > 1.3 || this.y > 1.9) {
      done = true;
    }

    return { done, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#06080e';
    ctx.fillRect(0, 0, w, h);

    // Estrelas
    ctx.fillStyle = '#64748b';
    for (let i = 0; i < 30; i++) {
      const sx = (Math.sin(i * 12.3) * 0.5 + 0.5) * w;
      const sy = (Math.cos(i * 7.7) * 0.5 + 0.5) * (h * 0.65);
      ctx.fillRect(sx, sy, 1.5, 1.5);
    }

    // Superfície Lunar & Pad
    const groundScreenY = h - 55;
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(0, groundScreenY + 15);
    ctx.lineTo(w * 0.35, groundScreenY + 5);
    ctx.lineTo(w * 0.4, groundScreenY);
    ctx.lineTo(w * 0.6, groundScreenY);
    ctx.lineTo(w * 0.65, groundScreenY + 8);
    ctx.lineTo(w, groundScreenY + 12);
    ctx.lineTo(w, h);
    ctx.fill();

    // Plataforma de pouso verde
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(w * 0.38, groundScreenY);
    ctx.lineTo(w * 0.62, groundScreenY);
    ctx.stroke();

    // Bandeiras
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(w * 0.38, groundScreenY - 14, 2, 14);
    ctx.fillRect(w * 0.38 + 2, groundScreenY - 14, 8, 6);
    ctx.fillRect(w * 0.62, groundScreenY - 14, 2, 14);
    ctx.fillRect(w * 0.62 + 2, groundScreenY - 14, 8, 6);

    const toScreenX = (valX) => w / 2 + valX * (w * 0.42);
    const toScreenY = (valY) => groundScreenY - valY * (h * 0.45);

    for (const p of this.particles) {
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.beginPath();
      ctx.arc(toScreenX(p.x), toScreenY(p.y), 3 * p.life, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;

    const lx = toScreenX(this.x);
    const ly = toScreenY(this.y);

    ctx.save();
    ctx.translate(lx, ly);
    ctx.rotate(this.angle);

    // Pernas de pouso
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-10, 6);
    ctx.lineTo(-18, 18);
    ctx.lineTo(-24, 18);
    ctx.moveTo(10, 6);
    ctx.lineTo(18, 18);
    ctx.lineTo(24, 18);
    ctx.stroke();

    // Módulo Hexagonal
    ctx.fillStyle = this.isLanded ? '#22c55e' : this.isCrashed ? '#ef4444' : '#38bdf8';
    ctx.beginPath();
    ctx.moveTo(-12, 6);
    ctx.lineTo(12, 6);
    ctx.lineTo(16, -6);
    ctx.lineTo(8, -16);
    ctx.lineTo(-8, -16);
    ctx.lineTo(-16, -6);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Cockpit
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(0, -6, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.stroke();

    ctx.restore();

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | LUNARLANDER-V3', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    const alt = Math.max(0, (this.y - this.groundY) * 100).toFixed(0);
    const vel = (Math.hypot(this.vx, this.vy) * 10).toFixed(1);
    ctx.fillText(`Passos: ${this.steps} | Altitude: ${alt}m | Vel: ${vel}m/s | Combustível: ${this.fuel.toFixed(0)}%`, 20, 42);

    ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    if (this.isLanded) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: POUSO PERFEITO 🏆', 20, h - 16);
    } else if (this.isCrashed) {
      ctx.fillStyle = '#ef4444';
      ctx.fillText('STATUS: IMPACTO CRÍTICO 💥 (REINICIANDO)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const actionNames = ['ORBITANDO (NOOP)', 'PROPULSOR ESQUERDO ◀', 'PROPULSOR PRINCIPAL ▲', 'PROPULSOR DIREITO ▶'];
      ctx.fillText(`AÇÃO: ${actionNames[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// SIMULADOR MOUNTAINCAR-V0 (CONTROLE NÃO-LINEAR DE INÉRCIA EM JS)
// =====================================================================

class MountainCarSim {
  constructor() {
    this.minPosition = -1.2;
    this.maxPosition = 0.6;
    this.maxSpeed = 0.07;
    this.goalPosition = 0.5;
    this.power = 0.0015;
    this.gravity = 0.0025;
    this.position = -0.5;
    this.velocity = 0;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
    this.reset();
  }

  reset() {
    this.position = -0.6 + Math.random() * 0.2;
    this.velocity = 0;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
  }

  getState() {
    return new Float32Array([this.position, this.velocity]);
  }

  step(action) {
    this.lastAction = action;
    this.steps++;

    const force = action === 0 ? -1 : action === 2 ? 1 : 0;
    this.velocity += force * this.power + Math.cos(3 * this.position) * -this.gravity;
    this.velocity = Math.max(-this.maxSpeed, Math.min(this.maxSpeed, this.velocity));

    this.position += this.velocity;
    this.position = Math.max(this.minPosition, Math.min(this.maxPosition, this.position));

    if (this.position === this.minPosition && this.velocity < 0) {
      this.velocity = 0;
    }

    const done = this.position >= this.goalPosition;
    if (done) this.isGoalReached = true;

    return { done, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    const toScreenX = (pos) => {
      return ((pos - this.minPosition) / (this.maxPosition - this.minPosition)) * (w - 60) + 30;
    };

    const toScreenY = (pos) => {
      const heightNorm = Math.sin(3 * pos);
      return h - 70 - heightNorm * 80;
    };

    // Montanha
    ctx.beginPath();
    ctx.moveTo(30, h - 30);
    for (let px = this.minPosition; px <= this.maxPosition; px += 0.02) {
      ctx.lineTo(toScreenX(px), toScreenY(px));
    }
    ctx.lineTo(w - 30, h - 30);
    ctx.closePath();
    ctx.fillStyle = '#0f172a';
    ctx.fill();

    // Contorno
    ctx.beginPath();
    for (let px = this.minPosition; px <= this.maxPosition; px += 0.02) {
      const sx = toScreenX(px);
      const sy = toScreenY(px);
      if (px === this.minPosition) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bandeira da Meta
    const goalX = toScreenX(this.goalPosition);
    const goalY = toScreenY(this.goalPosition);

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(goalX, goalY);
    ctx.lineTo(goalX, goalY - 32);
    ctx.stroke();

    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.moveTo(goalX, goalY - 32);
    ctx.lineTo(goalX + 16, goalY - 24);
    ctx.lineTo(goalX, goalY - 16);
    ctx.closePath();
    ctx.fill();

    // Carro
    const carX = toScreenX(this.position);
    const carY = toScreenY(this.position);
    const slope = Math.cos(3 * this.position);
    const carAngle = Math.atan(slope);

    ctx.save();
    ctx.translate(carX, carY);
    ctx.rotate(carAngle);

    ctx.fillStyle = this.isGoalReached ? '#22c55e' : '#38bdf8';
    ctx.fillRect(-15, -12, 30, 10);

    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    ctx.arc(-10, -2, 4, 0, Math.PI * 2);
    ctx.arc(10, -2, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#facc15';
    ctx.fillRect(13, -10, 3, 5);

    ctx.restore();

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | MOUNTAINCAR-V0', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    const energyKin = (0.5 * this.velocity * this.velocity * 1000).toFixed(1);
    ctx.fillText(`Passos: ${this.steps} | Posição: ${this.position.toFixed(2)} | Inércia: ${energyKin}J`, 20, 42);

    ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    if (this.isGoalReached) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: OBJETIVO ALCANÇADO 🏆 (TOPO DO VALE)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const actNames = ['IMPULSO ESQUERDA ◀', 'EMBALO LIVRE (NEUTRO)', 'IMPULSO DIREITA ▶'];
      ctx.fillText(`AÇÃO: ${actNames[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// SIMULADOR ACROBOT-V1 (PÊNDULO DUPLO ACROBÁTICO EM JS)
// =====================================================================

class AcrobotSim {
  constructor() {
    this.th1 = 0;
    this.th2 = 0;
    this.dth1 = 0;
    this.dth2 = 0;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
    this.trail = [];
    this.reset();
  }

  reset() {
    this.th1 = (Math.random() - 0.5) * 0.2;
    this.th2 = (Math.random() - 0.5) * 0.2;
    this.dth1 = (Math.random() - 0.5) * 0.1;
    this.dth2 = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
    this.trail = [];
  }

  getState() {
    return new Float32Array([
      Math.cos(this.th1),
      Math.sin(this.th1),
      Math.cos(this.th2),
      Math.sin(this.th2),
      this.dth1,
      this.dth2,
    ]);
  }

  step(action) {
    this.lastAction = action;
    this.steps++;

    const dt = 0.05;
    const torque = action === 0 ? -1.0 : action === 2 ? 1.0 : 0.0;
    const g = 9.8;

    const dth1_acc = -g * Math.sin(this.th1) * 0.5 + torque * 0.4;
    const dth2_acc = -g * Math.sin(this.th1 + this.th2) * 0.5 - torque * 0.8;

    this.dth1 += dth1_acc * dt;
    this.dth2 += dth2_acc * dt;

    this.dth1 *= 0.995;
    this.dth2 *= 0.995;

    this.th1 += this.dth1 * dt;
    this.th2 += this.dth2 * dt;

    const tipHeight = -(Math.cos(this.th1) + Math.cos(this.th1 + this.th2));
    const done = tipHeight > 1.0;
    if (done) this.isGoalReached = true;

    return { done, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    const pivotX = w / 2;
    const pivotY = h / 2 + 30;
    const l1 = 55;
    const l2 = 55;

    const jointX = pivotX + l1 * Math.sin(this.th1);
    const jointY = pivotY + l1 * Math.cos(this.th1);

    const tipX = jointX + l2 * Math.sin(this.th1 + this.th2);
    const tipY = jointY + l2 * Math.cos(this.th1 + this.th2);

    this.trail.push({ x: tipX, y: tipY });
    if (this.trail.length > 25) this.trail.shift();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.trail.length; i++) {
      const p = this.trail[i];
      ctx.globalAlpha = ((i + 1) / this.trail.length) * 0.6;
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Linha de Meta
    const goalLineY = pivotY - l1;
    ctx.strokeStyle = '#f59e0b';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, goalLineY);
    ctx.lineTo(w - 40, goalLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText('META (SWING ACIMA DA LINHA)', 45, goalLineY - 6);

    // Primeiro braço
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(jointX, jointY);
    ctx.stroke();

    // Articulação
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(jointX, jointY, 6, 0, Math.PI * 2);
    ctx.fill();

    // Segundo braço
    ctx.strokeStyle = '#a3e635';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(jointX, jointY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Ponta final
    ctx.fillStyle = this.isGoalReached ? '#4ade80' : '#22c55e';
    ctx.beginPath();
    ctx.arc(tipX, tipY, 7, 0, Math.PI * 2);
    ctx.fill();

    // Pivô base
    ctx.fillStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | ACROBOT-V1', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText(`Passos Acrobáticos: ${this.steps}`, 20, 42);

    ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    if (this.isGoalReached) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: META ATINGIDA 🏆 (SWING COMPLETO)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const torques = ['TORQUE ANTI-HORÁRIO ◀', 'SEM TORQUE (LIVRE)', 'TORQUE HORÁRIO ▶'];
      ctx.fillText(`AÇÃO: ${torques[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// SIMULADOR ANT-V5 (ROBÓTICA QUADRUPEDAL CONTÍNUA MUJOCO EM JS)
// =====================================================================

class AntSim {
  constructor() {
    this.steps = 0;
    this.x = 0;
    this.y = 0;
    this.vx = 0;
    this.vy = 0;
    this.heading = 0;
    this.torsoHeight = 0.75;
    this.jointAngles = new Float32Array(8);
    this.trail = [];
    this.reset();
  }

  reset() {
    this.steps = 0;
    this.x = 0;
    this.y = 0;
    this.vx = 0;
    this.vy = 0;
    this.heading = 0;
    this.torsoHeight = 0.75;
    this.trail = [];
    for (let i = 0; i < 8; i++) {
      this.jointAngles[i] = (Math.random() - 0.5) * 0.3;
    }
  }

  getState() {
    const s = new Float32Array(27);
    s[0] = this.torsoHeight;
    s[1] = Math.cos(this.heading);
    s[2] = Math.sin(this.heading);
    for (let i = 0; i < 8; i++) s[3 + i] = this.jointAngles[i];
    s[11] = this.vx;
    s[12] = this.vy;
    s[13] = 0.0; // v_z
    for (let i = 14; i < 27; i++) s[i] = 0.05 * Math.sin(this.steps * 0.1 + i);
    return s;
  }

  step(action) {
    this.steps++;
    // Ação contínua (8 torques articulares) ou discreta
    let torques = new Float32Array(8);
    if (action && (Array.isArray(action) || action.length === 8)) {
      for (let i = 0; i < 8; i++) torques[i] = Math.max(-1.0, Math.min(1.0, action[i]));
    } else {
      const idx = typeof action === 'number' ? action : 0;
      for (let i = 0; i < 8; i++) {
        torques[i] = Math.sin((this.steps * 0.3) + (i * 0.785) + idx);
      }
    }

    for (let i = 0; i < 8; i++) {
      this.jointAngles[i] = Math.max(-1.0, Math.min(1.0, this.jointAngles[i] * 0.88 + torques[i] * 0.18));
    }

    const gaitEnergy = (Math.abs(this.jointAngles[0]) + Math.abs(this.jointAngles[2]) + Math.abs(this.jointAngles[4]) + Math.abs(this.jointAngles[6])) * 0.25;
    const forwardThrust = 0.08 + gaitEnergy * 0.15;
    this.heading += (this.jointAngles[0] - this.jointAngles[4]) * 0.05;

    this.vx = this.vx * 0.92 + forwardThrust * Math.cos(this.heading) * 0.2;
    this.vy = this.vy * 0.92 + forwardThrust * Math.sin(this.heading) * 0.2;

    this.x += this.vx;
    this.y += this.vy;

    return { done: this.steps >= 1000, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    // Grid MuJoCo do piso
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1;
    const gridSize = 30;
    const offsetX = (this.x * 20) % gridSize;
    const offsetY = (this.y * 20) % gridSize;
    for (let gx = -offsetX; gx < w; gx += gridSize) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
      ctx.stroke();
    }
    for (let gy = -offsetY; gy < h; gy += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }

    // Rastro da trajetória do torso
    const cx = w / 2;
    const cy = h / 2 + 10;

    this.trail.push({ x: cx - this.vx * 15, y: cy - this.vy * 15 });
    if (this.trail.length > 20) this.trail.shift();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.trail.length; i++) {
      ctx.globalAlpha = ((i + 1) / this.trail.length) * 0.5;
      if (i === 0) ctx.moveTo(this.trail[i].x, this.trail[i].y);
      else ctx.lineTo(this.trail[i].x, this.trail[i].y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Renderiza Robô Ant (Torso central + 4 Pernas articuladas)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.heading);

    // 4 Pernas articuladas
    const legAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
    for (let i = 0; i < 4; i++) {
      const baseAng = legAngles[i];
      const hipJ = this.jointAngles[i * 2] * 0.6;
      const ankleJ = this.jointAngles[i * 2 + 1] * 0.8;

      const hipLen = 28;
      const legLen = 32;

      const hipX = Math.cos(baseAng + hipJ) * hipLen;
      const hipY = Math.sin(baseAng + hipJ) * hipLen;

      const footX = hipX + Math.cos(baseAng + hipJ + ankleJ + 0.4) * legLen;
      const footY = hipY + Math.sin(baseAng + hipJ + ankleJ + 0.4) * legLen;

      // Coxa
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(hipX, hipY);
      ctx.stroke();

      // Articulação intermediária
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(hipX, hipY, 4, 0, Math.PI * 2);
      ctx.fill();

      // Canela/Pé
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(hipX, hipY);
      ctx.lineTo(footX, footY);
      ctx.stroke();

      // Extremidade do Pé
      ctx.fillStyle = '#22c55e';
      ctx.beginPath();
      ctx.arc(footX, footY, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Torso central esférico do Ant
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(0, 0, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Olhos / Direção frontal
    ctx.fillStyle = '#f43f5e';
    ctx.beginPath();
    ctx.arc(12, -6, 3.5, 0, Math.PI * 2);
    ctx.arc(12, 6, 3.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Cabeçalho
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | ANT-V5 (MUJOCO CONTÍNUO)', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    const speed = (Math.hypot(this.vx, this.vy) * 20).toFixed(2);
    ctx.fillText(`Passos: ${this.steps} | Velocidade: ${speed} m/s | 8 Juntas Ativas`, 20, 42);

    ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillStyle = '#4ade80';
    ctx.fillText(`STATUS: LOCOMOÇÃO CONTÍNUA EM FÍSICA MUJOCO 🐜`, 20, h - 16);
  }
}

// =====================================================================
// RUNTIME DO AGENTE SYSTEM 1 (ONNX RUNTIME WEB)
// =====================================================================

class System1AgentWeb {
  constructor(session) {
    this.session = session;
    let hxDim = 256;
    try {
      if (session && session.inputMetadata && session.inputMetadata.hx && session.inputMetadata.hx.dimensions) {
        const dims = session.inputMetadata.hx.dimensions;
        hxDim = Number(dims[dims.length - 1]) || 256;
      }
    } catch (_) {
      hxDim = 256;
    }
    this.hxDim = hxDim;
    this.hx = new Float32Array(this.hxDim);
    this.prevObs = null;
    this.initObs = null;
    this.prevAction = 0;
    this.prevReward = 0.0;
    this.dynamicCalibration = 0.0;
    this.stagnationCount = 0;
    this.rewardEma = 0.0;
    this.actionHistory = [];
  }

  resetMemory() {
    this.hx.fill(0.0);
    this.prevObs = null;
    this.initObs = null;
    this.prevAction = 0;
    this.prevReward = 0.0;
    this.dynamicCalibration = 0.0;
    this.stagnationCount = 0;
    this.rewardEma = 0.0;
    this.actionHistory = [];
  }

  async actWithConfidence(obsArr, prevReward = 0.0, options = {}) {
    const obsDim = obsArr.length;
    let deltaDim = obsDim;
    if (this.session && this.session.inputMetadata && this.session.inputMetadata.delta_obs) {
      const dDims = this.session.inputMetadata.delta_obs.dimensions;
      if (dDims && dDims.length >= 3 && Number(dDims[2]) >= 3 * obsDim) {
        deltaDim = 3 * obsDim;
      }
    } else if (this.meta && (this.meta.tri_compass || this.triCompass)) {
      deltaDim = 3 * obsDim;
    }

    if (!this.initObs) {
      this.initObs = new Float32Array(obsArr);
    }

    let deltaArr = new Float32Array(deltaDim);
    if (this.prevObs) {
      for (let i = 0; i < obsDim; i++) deltaArr[i] = obsArr[i] - this.prevObs[i];
    }
    if (deltaDim >= 3 * obsDim && this.initObs) {
      for (let i = 0; i < obsDim; i++) deltaArr[obsDim + i] = obsArr[i] - this.initObs[i];
    }
    this.prevObs = new Float32Array(obsArr);
    this.prevReward = typeof prevReward === 'number' ? prevReward : 0.0;

    const t0 = performance.now();

    const tObs = new ort.Tensor('float32', obsArr, [1, 1, obsDim]);
    const tDelta = new ort.Tensor('float32', deltaArr, [1, 1, deltaDim]);
    const tAct = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.prevAction)]), [1, 1]);
    const tRew = new ort.Tensor('float32', new Float32Array([this.prevReward]), [1, 1, 1]);
    const tHx = new ort.Tensor('float32', this.hx, [1, 1, this.hxDim]);

    const feeds = {
      obs: tObs,
      delta_obs: tDelta,
      prev_action: tAct,
      prev_reward: tRew,
      hx: tHx,
    };

    const results = await this.session.run(feeds);
    const latencyMs = performance.now() - t0;

    const logits = results.logits.data;
    if (results.next_hx && results.next_hx.data) {
      if (this.hx.length !== results.next_hx.data.length) {
        this.hxDim = results.next_hx.data.length;
        this.hx = new Float32Array(this.hxDim);
      }
      this.hx.set(results.next_hx.data);
    }

    // Detecção de Action Chunking Temporal (ACT) nos logits
    let k = 1;
    let nActs = logits.length;
    if (this.actionChunkSize && this.actionChunkSize > 1) {
      k = this.actionChunkSize;
      nActs = Math.floor(logits.length / k);
    } else if (results.logits && results.logits.dims && results.logits.dims.length === 3) {
      k = results.logits.dims[1];
      nActs = results.logits.dims[2];
    } else if (logits.length > 12 && logits.length % 12 === 0) {
      k = logits.length / 12;
      nActs = 12;
    }

    const chunkActions = [];
    if (k > 1) {
      for (let s = 0; s < k; s++) {
        let maxSub = -Infinity;
        let bestSub = 0;
        const offset = s * nActs;
        for (let a = 0; a < nActs; a++) {
          const val = logits[offset + a];
          if (val > maxSub) { maxSub = val; bestSub = a; }
        }
        chunkActions.push(bestSub);
      }
    }

    // Softmax & Entropia de Shannon no primeiro passo do chunk (Gating nos logits originais)
    let maxLogit = -Infinity;
    for (let i = 0; i < nActs; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const probs = new Float32Array(nActs);
    for (let i = 0; i < nActs; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    let entropy = 0;
    let bestAction = 0;
    let maxProb = 0;
    const ranked = [];
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
      ranked.push({ action: i, prob: probs[i], logit: logits[i] });
      if (probs[i] > maxProb) { maxProb = probs[i]; bestAction = i; }
      if (probs[i] > 1e-9) entropy -= probs[i] * Math.log(probs[i]);
    }
    ranked.sort((a, b) => b.prob - a.prob);

    const maxEntropy = Math.log(nActs);
    const uncertainty = Math.min(1.0, Math.max(0.0, entropy / maxEntropy));

    // Calibração Contínua de Ação [0.0 = Determinístico/argmax, 1.0 = Estocástico Total] ou Auto-Calibração Homeostática
    let calib = 0.5;
    const isAuto = Boolean(options.autoCalibrate || options.calibration === 'auto');
    if (isAuto) {
      const r = this.prevReward;
      const deltaR = r - this.rewardEma;
      this.rewardEma = 0.9 * this.rewardEma + 0.1 * r;

      if (r > 0.01 || deltaR > 0.01) {
        this.stagnationCount = 0;
        this.dynamicCalibration = Math.max(0.0, this.dynamicCalibration - 0.20);
      } else {
        this.stagnationCount++;
        if (this.stagnationCount >= 2) {
          this.dynamicCalibration = Math.min(0.80, this.dynamicCalibration + 0.10);
        }
      }
      if (this.dynamicCalibration <= 0.0) {
        calib = 0.0;
      } else {
        calib = Math.min(0.80, Math.max(0.0, this.dynamicCalibration * (0.6 + 0.4 * uncertainty) + 0.10 * Math.max(0.0, uncertainty - 0.50)));
      }
    } else {
      const calibration = options.calibration !== undefined 
        ? options.calibration 
        : (options.temperature !== undefined ? options.temperature : (options.deterministic ? 0.0 : 0.5));
      calib = Math.min(1.0, Math.max(0.0, typeof calibration === 'number' ? calibration : 0.5));
    }

    // Resolução do conjunto de ações a evitar / quebra de ciclos e qualidade
    const avoidSet = new Set();
    if (options.avoidAction !== undefined) {
      if (Array.isArray(options.avoidAction)) {
        options.avoidAction.forEach(a => { if (typeof a === 'number' && a >= 0) avoidSet.add(a); });
      } else if (typeof options.avoidAction === 'number' && options.avoidAction >= 0) {
        avoidSet.add(options.avoidAction);
      }
    }

    const hist = this.actionHistory || [];
    if ((isAuto && this.stagnationCount >= 2) || (options.qualityFilter !== false && hist.length >= 3)) {
      if (hist.length >= 3 && hist[hist.length - 1] === hist[hist.length - 3]) {
        // Oscilação A -> B -> A sob estagnação: evita B para romper o ciclo vicioso
        avoidSet.add(hist[hist.length - 2]);
      } else if (hist.length >= 3 && hist[hist.length - 1] === hist[hist.length - 2] && hist[hist.length - 2] === hist[hist.length - 3]) {
        avoidSet.add(hist[hist.length - 1]);
      }
    }
    // Action Quality Filter & Inverse Pruning:
    // Em cubo atômico (12 ações), o movimento imediatamente inverso (ex: U -> U')
    // anula o passo anterior e desperdiça tempo. Evita reversão direta se houver histórico e maxProb < 0.96.
    if (options.qualityFilter !== false && hist.length >= 1 && logits.length === 12) {
      const last = hist[hist.length - 1];
      const inv = (last % 2 === 0) ? last + 1 : last - 1;
      if (maxProb < 0.96) {
        avoidSet.add(inv);
      }
    }

    let chosenAction = bestAction;
    if (calib > 0.01) {
      // Inovação proporcional com probabilidade = calib
      if (Math.random() < calib) {
        const T = 1.0 + 0.5 * calib;
        let maxScaled = -Infinity;
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] / T > maxScaled) maxScaled = logits[i] / T;
        }
        let sumExpT = 0;
        const sampledProbs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
          sampledProbs[i] = Math.exp((logits[i] / T) - maxScaled);
          sumExpT += sampledProbs[i];
        }
        for (let i = 0; i < logits.length; i++) sampledProbs[i] /= sumExpT;

        const randVal = Math.random();
        let cum = 0;
        for (let i = 0; i < sampledProbs.length; i++) {
          cum += sampledProbs[i];
          if (randVal <= cum || i === sampledProbs.length - 1) {
            chosenAction = i;
            break;
          }
        }
      } else {
        chosenAction = bestAction;
      }
    }

    // Quebra estrita de ciclos (Cycle Breaker):
    if (avoidSet.size > 0 && avoidSet.has(chosenAction) && ranked.length > 1) {
      const alt = ranked.find(item => !avoidSet.has(item.action));
      chosenAction = alt ? alt.action : ranked[1].action;
    }

    if (!this.actionHistory) this.actionHistory = [];
    this.actionHistory.push(chosenAction);
    if (this.actionHistory.length > 8) this.actionHistory.shift();

    if (chunkActions.length > 0) {
      chunkActions[0] = chosenAction;
    }

    return {
      action: chosenAction,
      chunk: chunkActions.length > 0 ? chunkActions : [chosenAction],
      actionChunkSize: k,
      confidence: maxProb,
      uncertainty: uncertainty,
      calibration: calib,
      isUncertain: uncertainty > 0.70 || maxProb < 0.50,
      latencyMs: latencyMs,
      ranked: ranked,
      value: results.value ? results.value.data[0] : 0.0
    };
  }
}

// Expõe globalmente para uso no portfólio
window.System1 = {
  RubiksCubeSim,
  CartPoleSim,
  LunarLanderSim,
  MountainCarSim,
  AcrobotSim,
  AntSim,
  System1AgentWeb,
};
