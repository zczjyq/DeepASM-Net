class Simulation {
    constructor() {
        this.particles = [];
        this.particleEmitters = [];

        this.AMOUNT_PARTICLES = 2000;      // 模拟的粒子总数
        this.VELOCITY_DAMPING = 1;         // 速度阻尼（1表示无能量损耗，值越小流体越粘稠）
        this.GRAVITY = new Vector2(0, 1);  // 重力向量（向下）
        this.REST_DENSITY = 10;            // 静止密度（流体期望维持的基准密度）
        this.K_NEAR = 3;                   // 近距离压力系数（控制粒子极近时的强排斥力，防止重叠）
        this.K = 0.2;                     // 普通压力系数（控制整体不可压缩性）
        this.INTERACTION_RADIUS = 25;      // 相互作用半径（也叫平滑核半径 h，粒子只与此范围内的邻居发生作用）

        // 相互作用半径
        this.SIGMA = 0.1;
        this.BETA = 0.02;

        this.fluidHashGrid = new FluidHashGrid(this.INTERACTION_RADIUS);
        this.rotate = false;


        // this.instantiateParticeles();
        this.fluidHashGrid.initialize(this.particles);

        this.emitter = this.createParticleEmitter(
            new Vector2(canvas.width / 2, 400), // position
            new Vector2(0, -1),
            30,
            0.5,
            8,
            20,
        )
    }

    createParticleEmitter(position, direction, size, spawnInterval, amount, velocity) {
        let emitter = new ParticleEmitter(position, direction, size, spawnInterval, amount, velocity);
        this.particleEmitters.push(emitter);
        return emitter;
    }


    neighbourSearch(mousePos) {
        this.fluidHashGrid.clearGrid();
        this.fluidHashGrid.mapParticleToCell();

        // this.particles[0].position = mousePos.Cpy();        
        // let contentOfCell = this.fluidHashGrid.getNeighbourOfParticleIdx(0);

        // for (let i = 0; i < this.particles.length; i++)
        //     this.particles[i].color = "#28beff";
        // for (let i = 0; i < contentOfCell.length; i++) {

        //     let particle = contentOfCell[i];
        //     let direction = Sub(particle.position, mousePos);
        //     let distanceSquared = direction.Length2();
        //     if (distanceSquared <= this.NEIGHBOUR_SEARCH_RADIUS * this.NEIGHBOUR_SEARCH_RADIUS) {
        //         particle.color = "orange";
        //     }
        // }
    }

    update(dt) {
        this.emitter.spawn(dt, this.particles)
        if (this.rotate) {
            this.emitter.rotate(0.01);
        }
        this.applyGravity(dt);

        this.viccosity(dt);

        this.predictPositions(dt);

        this.neighbourSearch();

        this.doubleDensityRealxation(dt);

        this.worldBoundart();

        this.computeNextVelcity(dt);


    }

    viccosity(dt) {
        for (let i = 0; i < this.particles.length; i++) {
            let neighbours = this.fluidHashGrid.getNeighbourOfParticleIdx(i);
            let particleA = this.particles[i];

            for (let j = 0; j < neighbours.length; j++) {
                let particleB = neighbours[j];
                if (particleA === particleB) continue;

                let rij = Sub(particleB.position, particleA.position);
                let velocityA = particleA.velocity;
                let velocityB = particleB.velocity;

                let q = rij.Length() / this.INTERACTION_RADIUS;

                if (q < 1) {
                    rij.Normalize();
                    let u = Sub(velocityA, velocityB).Dot(rij);

                    if (u > 0) {
                        let ITerm = dt * (1 - q) * (this.SIGMA * u + this.BETA * u * u);
                        let I = Scale(rij, ITerm);
                        

                        particleA.velocity = Sub(particleA.velocity, Scale(I, 0.5));
                        particleB.velocity = Add(particleB.velocity, Scale(I, 0.5));
                    }
                }
            }
        }
    }

    doubleDensityRealxation(dt) {
        for (let i = 0; i < this.particles.length; i++) {

            let density = 0;
            let densityNear = 0;
            let neighbours = this.fluidHashGrid.getNeighbourOfParticleIdx(i);
            let particleA = this.particles[i];
            // console.log(particleA);

            for (let j = 0; j < neighbours.length; j++) {
                let particleB = neighbours[j];
                if (particleA === particleB) continue;



                let rij = Sub(particleB.position, particleA.position);


                let q = rij.Length() / this.INTERACTION_RADIUS;

                if (q < 1.0) {
                    density += Math.pow(1 - q, 2);
                    densityNear += Math.pow(1 - q, 3);
                }
            }

            let pressure = this.K * (density - this.REST_DENSITY);
            let pressureNear = this.K_NEAR * densityNear;
            let particleDisplacement = Vector2.Zero();
            // console.log(particleA);
            for (let j = 0; j < neighbours.length; j++) {
                let particleB = neighbours[j];
                if (particleA === particleB) continue;


                let rij = Sub(particleB.position, particleA.position);
                let q = rij.Length() / this.INTERACTION_RADIUS;
                if (q < 1.0) {
                    rij.Normalize();
                    let displacementTerm = Math.pow(dt, 2) * (pressure * (1 - q) + pressureNear * Math.pow(1 - q, 2));
                    let D = Scale(rij, displacementTerm);

                    particleB.position = Add(particleB.position, Scale(D, 0.5));
                    particleDisplacement = Sub(particleDisplacement, Scale(D, 0.5));


                }
            }
            particleA.position = Add(particleA.position, particleDisplacement);

        }
    }

    applyGravity(dt) {
        for (let i = 0; i < this.particles.length; i++) {
            this.particles[i].velocity = Add(this.particles[i].velocity, Scale(this.GRAVITY, dt));
        }
    }

    // 生产粒子
    instantiateParticeles() {
        let offsetBetweenParticles = 10;
        let offsetAllParticles = new Vector2(750, 100);
        let xParticles = Math.sqrt(this.AMOUNT_PARTICLES);
        let yParticles = xParticles;
        for (let x = 0; x < xParticles; x++) {
            for (let y = 0; y < yParticles; y++) {
                let position = new Vector2(x * offsetBetweenParticles, y * offsetBetweenParticles);
                position = Add(position, offsetAllParticles);

                let particle = new Particle(position);
                // particle.velocity = Scale(new Vector2(-0.5 + Math.random(), -0.5 + Math.random()), 1000);

                this.particles.push(particle);
            }
        }
    }

    predictPositions(dt) {
        for (let i = 0; i < this.particles.length; i++) {
            this.particles[i].prePosition = this.particles[i].position.Cpy()
            let positionDelta = Scale(this.particles[i].velocity, dt * this.VELOCITY_DAMPING);
            this.particles[i].position = Add(this.particles[i].position, positionDelta);
        }
        // console.log(this.particles);
    }

    computeNextVelcity(dt) {
        for (let i = 0; i < this.particles.length; i++) {
            let velocity = Scale(Sub(this.particles[i].position, this.particles[i].prePosition), 1.0 / dt);
            this.particles[i].velocity = velocity;
        }
    }

    worldBoundart() {
        for (let i = 0; i < this.particles.length; i++) {
            let pos = this.particles[i].position;
            let prevPos = this.particles[i].prePosition;
            if (pos.x < 0) {
                this.particles[i].position.x = 0;
                this.particles[i].prePosition.x = 0;
            }
            if (pos.x > canvas.width) {
                this.particles[i].position.x = canvas.width;
                this.particles[i].prePosition.x = canvas.width;
            }
            if (pos.y < 0) {
                this.particles[i].position.y = 0;
                this.particles[i].prePosition.y = 0;
            }
            if (pos.y > canvas.height) {
                this.particles[i].position.y = canvas.height;
                this.particles[i].prePosition.y = canvas.height;
            }

        }
    }

    // 绘画粒子
    draw() {
        for (let i = 0; i < this.particles.length; i++) {
            let position = this.particles[i].position;
            let color = this.particles[i].color;
            DrawUtils.drawPoint(position, 5, color);
        }

        for (let i = 0; i < this.particleEmitters.length; i++) {
            this.particleEmitters[i].draw();
        }
    }
}