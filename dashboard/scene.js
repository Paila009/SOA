/* A small, dependency-free orbital scene. The chat can remove and restore it. */
(() => {
  "use strict";

  const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const TAU = Math.PI * 2;
  let activeScene = null;
  let reconcileQueued = false;

  function createScene(canvas) {
    const context = canvas.getContext("2d", { alpha: true });
    const stage = canvas.parentElement;
    if (!context || !stage) return null;

    let width = 0;
    let height = 0;
    let frame = 0;
    let lastFrame = 0;
    let elapsed = 0;
    let visible = true;
    let disposed = false;
    let pointerX = 0;
    let pointerY = 0;
    let easedX = 0;
    let easedY = 0;
    const stars = Array.from({ length: 31 }, (_, index) => ({
      x: ((index * 0.61803398875 + 0.13) % 1),
      y: ((index * 0.41421356237 + 0.21) % 1),
      size: index % 5 === 0 ? 1.35 : 0.7,
      phase: index * 1.73,
    }));

    function project(point, rotation, radius, centerX, centerY) {
      const yaw = rotation + easedX * 0.12;
      const pitch = -0.22 + easedY * 0.1;
      const x = point[0] * Math.cos(yaw) + point[2] * Math.sin(yaw);
      const z = -point[0] * Math.sin(yaw) + point[2] * Math.cos(yaw);
      const y = point[1] * Math.cos(pitch) - z * Math.sin(pitch);
      const depth = point[1] * Math.sin(pitch) + z * Math.cos(pitch);
      const perspective = 5.5 / (5.5 - depth);
      return {
        x: centerX + x * radius * perspective,
        y: centerY + y * radius * perspective,
        depth,
        scale: perspective,
      };
    }

    function orbitPoint(angle, tilt, roll, radius) {
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius * Math.cos(tilt);
      const z = Math.sin(angle) * radius * Math.sin(tilt);
      return [x * Math.cos(roll) - y * Math.sin(roll), x * Math.sin(roll) + y * Math.cos(roll), z];
    }

    function glow(x, y, radius, rgb, strength) {
      const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, `rgba(${rgb},${strength})`);
      gradient.addColorStop(0.3, `rgba(${rgb},${strength * 0.3})`);
      gradient.addColorStop(1, `rgba(${rgb},0)`);
      context.fillStyle = gradient;
      context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    }

    function render() {
      if (disposed || !width || !height) return;
      const time = motionPreference.matches ? 0 : elapsed * 0.00015;
      const radius = Math.min(width / 5.3, height / 3.55, 84);
      const centerX = width * 0.5 + easedX * 5;
      const centerY = height * 0.51 + easedY * 3;
      const rotation = time * 0.52 + 0.45;
      const point = (value) => project(value, rotation, radius, centerX, centerY);
      context.clearRect(0, 0, width, height);

      glow(centerX - radius * 0.3, centerY, radius * 2.7, "35,108,214", 0.13);
      glow(centerX + radius * 0.6, centerY - radius * 0.25, radius * 1.9, "93,78,224", 0.09);

      stars.forEach((star) => {
        const opacity = 0.13 + (Math.sin(time * 1.4 + star.phase) + 1) * 0.1;
        context.fillStyle = `rgba(135,200,235,${opacity})`;
        context.beginPath();
        context.arc(width * (0.15 + star.x * 0.7), height * (0.1 + star.y * 0.8), star.size, 0, TAU);
        context.fill();
      });

      // A grounded shadow gives the suspended object a little physical depth.
      context.save();
      context.translate(centerX, centerY + radius * 1.3);
      context.scale(1, 0.15);
      glow(0, 0, radius * 1.35, "21,84,161", 0.16);
      context.restore();

      const orbits = [
        { tilt: 1.1, roll: -0.3, radius: 1.79, color: "93,226,250", opacity: 0.66, speed: 0.8, phase: 0.6 },
        { tilt: 0.96, roll: 0.73, radius: 1.54, color: "137,127,255", opacity: 0.49, speed: -0.58, phase: 3.7 },
        { tilt: 1.2, roll: -0.95, radius: 1.41, color: "112,180,246", opacity: 0.27, speed: 0.47, phase: 2.1 },
      ];

      function drawOrbits(front) {
        orbits.forEach((orbit, orbitIndex) => {
          const segments = 128;
          for (let index = 0; index < segments; index += 1) {
            const a = point(orbitPoint(index / segments * TAU, orbit.tilt, orbit.roll, orbit.radius));
            const b = point(orbitPoint((index + 1) / segments * TAU, orbit.tilt, orbit.roll, orbit.radius));
            if ((a.depth > 0) !== front) continue;
            context.strokeStyle = `rgba(${orbit.color},${orbit.opacity * (front ? 0.78 : 0.27)})`;
            context.lineWidth = orbitIndex === 0 ? 1.15 : 0.8;
            context.beginPath();
            context.moveTo(a.x, a.y);
            context.lineTo(b.x, b.y);
            context.stroke();
          }
          const satellite = point(orbitPoint(time * orbit.speed + orbit.phase, orbit.tilt, orbit.roll, orbit.radius));
          if ((satellite.depth > 0) !== front) return;
          glow(satellite.x, satellite.y, 17 * satellite.scale, orbit.color, front ? 0.36 : 0.14);
          context.fillStyle = front ? "#d2f6ff" : `rgba(${orbit.color},0.56)`;
          context.beginPath();
          context.arc(satellite.x, satellite.y, (orbitIndex === 0 ? 3.1 : 2.2) * satellite.scale, 0, TAU);
          context.fill();
          context.strokeStyle = `rgba(${orbit.color},0.2)`;
          context.lineWidth = 0.8;
          context.beginPath();
          context.arc(satellite.x, satellite.y, 6.8 * satellite.scale, 0, TAU);
          context.stroke();
        });
      }

      drawOrbits(false);

      // A glass-like core, with light falling from the upper-left edge.
      glow(centerX, centerY, radius * 1.24, "54,153,231", 0.2);
      const surface = context.createRadialGradient(
        centerX - radius * 0.42, centerY - radius * 0.49, radius * 0.05,
        centerX + radius * 0.14, centerY + radius * 0.13, radius * 1.09,
      );
      surface.addColorStop(0, "rgba(43,96,139,0.92)");
      surface.addColorStop(0.34, "rgba(18,47,84,0.96)");
      surface.addColorStop(0.76, "rgba(10,23,48,0.97)");
      surface.addColorStop(1, "rgba(47,62,128,0.91)");
      context.fillStyle = surface;
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, TAU);
      context.fill();

      context.save();
      context.beginPath();
      context.arc(centerX, centerY, radius - 0.5, 0, TAU);
      context.clip();

      function meshLine(values) {
        for (let index = 0; index < values.length - 1; index += 1) {
          const a = point(values[index]);
          const b = point(values[index + 1]);
          if (a.depth < 0) continue;
          context.strokeStyle = `rgba(118,213,241,${0.06 + a.depth * 0.17})`;
          context.lineWidth = 0.65;
          context.beginPath();
          context.moveTo(a.x, a.y);
          context.lineTo(b.x, b.y);
          context.stroke();
        }
      }

      for (let latitude = -3; latitude <= 3; latitude += 1) {
        const phi = latitude * Math.PI / 9;
        const ring = [];
        for (let step = 0; step <= 64; step += 1) {
          const theta = step / 64 * TAU;
          ring.push([Math.cos(phi) * Math.cos(theta), Math.sin(phi), Math.cos(phi) * Math.sin(theta)]);
        }
        meshLine(ring);
      }
      for (let longitude = 0; longitude < 12; longitude += 1) {
        const theta = longitude / 12 * TAU;
        const meridian = [];
        for (let step = 0; step <= 40; step += 1) {
          const phi = -Math.PI / 2 + step / 40 * Math.PI;
          meridian.push([Math.cos(phi) * Math.cos(theta), Math.sin(phi), Math.cos(phi) * Math.sin(theta)]);
        }
        meshLine(meridian);
      }

      // Distributed signal nodes move with the sphere, rather than over it.
      for (let index = 0; index < 22; index += 1) {
        const y = 1 - (index + 0.5) / 22 * 2;
        const angle = index * 2.39996323;
        const spread = Math.sqrt(1 - y * y);
        const node = point([Math.cos(angle) * spread, y, Math.sin(angle) * spread]);
        if (node.depth <= 0.03) continue;
        const pulse = 0.55 + 0.45 * Math.sin(time * 2 + index);
        glow(node.x, node.y, 7 + pulse * 4, "109,232,249", 0.1 + pulse * 0.08);
        context.fillStyle = `rgba(166,244,255,${0.4 + node.depth * 0.5})`;
        context.beginPath();
        context.arc(node.x, node.y, (index % 4 === 0 ? 1.8 : 1.05) * node.scale, 0, TAU);
        context.fill();
      }
      glow(centerX - radius * 0.55, centerY - radius * 0.6, radius * 0.62, "87,212,246", 0.12);
      context.restore();

      const edge = context.createLinearGradient(centerX - radius, centerY - radius, centerX + radius, centerY + radius);
      edge.addColorStop(0, "rgba(163,239,255,0.75)");
      edge.addColorStop(0.36, "rgba(73,161,222,0.16)");
      edge.addColorStop(0.72, "rgba(100,112,241,0.13)");
      edge.addColorStop(1, "rgba(140,139,253,0.58)");
      context.strokeStyle = edge;
      context.lineWidth = 1.15;
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, TAU);
      context.stroke();
      drawOrbits(true);
    }

    function canAnimate() {
      return !disposed && canvas.isConnected && visible && !document.hidden && !motionPreference.matches;
    }

    function tick(timestamp) {
      frame = 0;
      if (!canAnimate()) return;
      if (!lastFrame || timestamp - lastFrame >= 32) {
        const delta = lastFrame ? Math.min(timestamp - lastFrame, 65) : 33;
        elapsed += delta;
        lastFrame = timestamp;
        easedX += (pointerX - easedX) * 0.075;
        easedY += (pointerY - easedY) * 0.075;
        render();
      }
      frame = requestAnimationFrame(tick);
    }

    function refresh() {
      if (disposed) return;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      lastFrame = 0;
      if (motionPreference.matches) easedX = easedY = 0;
      render();
      if (canAnimate()) frame = requestAnimationFrame(tick);
    }

    function resize() {
      if (disposed) return;
      const bounds = stage.getBoundingClientRect();
      width = Math.max(0, bounds.width);
      height = Math.max(0, bounds.height);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      refresh();
    }

    function movePointer(event) {
      if (motionPreference.matches || event.pointerType === "touch") return;
      const bounds = stage.getBoundingClientRect();
      pointerX = Math.max(-1, Math.min(1, (event.clientX - bounds.left) / Math.max(bounds.width, 1) * 2 - 1));
      pointerY = Math.max(-1, Math.min(1, (event.clientY - bounds.top) / Math.max(bounds.height, 1) * 2 - 1));
    }

    function resetPointer() {
      pointerX = pointerY = 0;
    }

    const resizeObserver = new ResizeObserver(resize);
    const intersectionObserver = new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      refresh();
    }, { threshold: 0 });
    resizeObserver.observe(stage);
    intersectionObserver.observe(canvas);
    stage.addEventListener("pointermove", movePointer, { passive: true });
    stage.addEventListener("pointerleave", resetPointer, { passive: true });
    resize();

    return {
      canvas,
      refresh,
      dispose() {
        disposed = true;
        if (frame) cancelAnimationFrame(frame);
        resizeObserver.disconnect();
        intersectionObserver.disconnect();
        stage.removeEventListener("pointermove", movePointer);
        stage.removeEventListener("pointerleave", resetPointer);
      },
    };
  }

  function reconcile() {
    reconcileQueued = false;
    const canvas = document.getElementById("signal-scene");
    if (activeScene && activeScene.canvas === canvas) return;
    if (activeScene) activeScene.dispose();
    activeScene = canvas instanceof HTMLCanvasElement ? createScene(canvas) : null;
  }

  function initialize() {
    reconcile();
    // New conversation restores #welcome with innerHTML, producing a new canvas.
    new MutationObserver(() => {
      if (reconcileQueued) return;
      reconcileQueued = true;
      queueMicrotask(reconcile);
    }).observe(document.body, { childList: true, subtree: true });
    document.addEventListener("visibilitychange", () => activeScene?.refresh());
    motionPreference.addEventListener("change", () => activeScene?.refresh());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
