// ----------------------------------------------------------------------------

import { mat4, vec3 } from 'https://cdn.skypack.dev/gl-matrix';

// ----------------------------------------------------------------------------

export class ViewController {
  update(dt) {
  }

  getViewMatrix(viewMatrix) {
  }

  target() {
    return vec3.fromValues(0.0, 0.0, 0.0);
  }
}

// ----------------------------------------------------------------------------

export class ArcBallController extends ViewController {
  constructor() {
    super();
    this.yaw = 0.0;
    this.yaw2 = 0.0;
    this.pitch = 0.0;
    this.pitch2 = 0.0;
    this.dolly = 0.0;
    this.dolly2 = 0.0;
    this.target = vec3.fromValues(0.0, 0.0, 0.0);
    this.target2 = vec3.fromValues(0.0, 0.0, 0.0);
    this.rotationMatrix = mat4.create();
    this.sideViewSet = false;

    // Constants
    this.rotateEpsilon = 1e-7;
    this.angleModulo = Math.PI * 2;
    this.mouseRAcceleration = 0.004;
    this.mouseTAcceleration = 0.002;
    this.mouseWAcceleration = 0.15;
    this.smoothingCoeff = 12.0;

    // ---

    this.btnRotate = (e) => (e.button === 2)
                         || ((e.pointerType === 'touch') && (e.pointerId))
                         ;
    this.btnTranslate = (e) => false;
    this.isRotating = false;
    this.isTranslating = false;

    // ----
    this.mouseMoved = false;
    this.mouseX = 0;
    this.mouseY = 0;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.wheelDelta = 0;
    // ----
  }

  setEvents(canvas) {
    canvas.addEventListener('pointerdown', (e) => {
      this.isRotating = this.btnRotate(e);
      this.isTranslating = this.btnTranslate(e);
    });

    canvas.addEventListener('pointerup', (e) => {
      this.isRotating = false;
      this.isTranslating = false;
    });

    canvas.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'mouse') {
        this.mouseMoved = true;
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
      }
    });

    canvas.addEventListener('wheel', (e) => {
      this.wheelDelta = e.deltaY;
    });
  }

  update(dt) {
    if (this.mouseMoved) {
      this.handleMouseMove(this.isTranslating, this.isRotating, this.mouseX, this.mouseY);
    }

    this.handleMouseWheel(this.wheelDelta);
    this.smoothTransition(dt);
    this.regulateAngle(this.pitch, this.pitch2);
    this.regulateAngle(this.yaw, this.yaw2);

    this.wheelDelta = 0;
    this.mouseMoved = false;
  }

  getViewMatrix(viewMatrix) {
    const eye = vec3.fromValues(
      Math.cos(this.yaw) * this.dolly,
      Math.sin(this.pitch) * this.dolly,
      Math.sin(this.yaw) * this.dolly
    );
    const up = vec3.fromValues(0, 1, 0);
    mat4.lookAt(viewMatrix, eye, this.target, up);
  }

  handleMouseMove(translating, rotating, mouseX, mouseY) {
    const deltaX = mouseX - this.lastMouseX;
    const deltaY = mouseY - this.lastMouseY;
    this.lastMouseX = mouseX;
    this.lastMouseY = mouseY;

    if ((Math.abs(deltaX) + Math.abs(deltaY)) < this.rotateEpsilon) {
      return;
    }

    if (translating) {
      const acc = this.dolly2 * this.mouseTAcceleration;
      const translation = vec3.fromValues(deltaX * acc, -deltaY * acc, 0);
      vec3.transformMat4(translation, translation, this.rotationMatrix);
      vec3.add(this.target, this.target, translation);
      vec3.copy(this.target2, this.target);
    }

    if (rotating) {
      this.pitch2 += deltaY * this.mouseRAcceleration;
      this.yaw2 += deltaX * this.mouseRAcceleration;
      this.sideViewSet = false;
    }
  }

  handleMouseWheel(wheelDelta) {
    const sign =  ((Math.abs(wheelDelta) > 1e-5) ? (wheelDelta > 0 ? 1 : -1) : 0);
    this.dolly2 *= 1.0 + sign * this.mouseWAcceleration;
  }

  smoothTransition(dt) {
    const k = Math.min(this.smoothingCoeff * dt, 1.0);
    this.yaw = this.lerp(this.yaw, this.yaw2, k);
    this.pitch = this.lerp(this.pitch, this.pitch2, k);
    this.dolly = this.lerp(this.dolly, this.dolly2, k);
    vec3.lerp(this.target, this.target, this.target2, k);
  }

  lerp(a, b, t) {
    return a + t * (b - a);
  }

  regulateAngle(current, target) {
    if (Math.abs(target) >= this.angleModulo) {
      const dist = target - current;
      target = target % this.angleModulo;
      current = target - dist;
    }
  }

  setView(yaw, pitch, smooth = true) {
    this.setYaw(yaw, smooth);
    this.setPitch(pitch, smooth);
  }

  setYaw(value, smooth = true) {
    this.yaw2 = value;
    if (!smooth) {
      this.yaw = value;
    }
  }

  setPitch(value, smooth = true) {
    this.pitch2 = value;
    if (!smooth) {
      this.pitch = value;
    }
  }

  setDolly(value, smooth = true) {
    this.dolly2 = value;
    if (!smooth) {
      this.dolly = value;
    }
  }

  resetTarget() {
    vec3.set(this.target, 0.0, 0.0, 0.0);
    vec3.copy(this.target2, this.target);
  }
}

// ----------------------------------------------------------------------------