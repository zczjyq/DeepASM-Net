/**
 * 流体模拟哈希网格类
 * 用于将粒子映射到固定尺寸的网格单元格中，通过哈希快速索引单元格，
 * 高效查询邻域粒子，提升流体模拟中粒子交互计算的性能
 */
class FluidHashGrid {
    /**
     * 构造函数
     * @param {number} cellSize - 网格单元格的边长（需适配粒子半径）
     */
    constructor(cellSize) {
        // 网格单元格的边长
        this.cellSize = cellSize;
        // 哈希映射表：key=单元格哈希值，value=该单元格内的粒子数组
        this.hashMap = new Map();
        // 哈希表最大尺寸（取模用，控制哈希值范围，减少冲突）
        this.hashMapSize = 10000000;
        // 哈希计算用的大质数1（降低哈希冲突概率）
        this.prime1 = 6614058611;
        // 哈希计算用的大质数2（配合prime1进一步减少冲突）
        this.prime2 = 7528850467;
        // 存储所有待映射的流体粒子
        this.particles = [];
    }

    /**
     * 初始化网格的粒子数据源
     * @param {Array} particles - 流体粒子数组（每个粒子需包含position属性，且position有x、y坐标）
     */
    initialize(particles) {
        this.particles = particles;
    }

    /**
     * 清空哈希网格的映射关系
     * 每帧更新粒子位置前需调用，清除旧的粒子-单元格映射
     */
    clearGrid() {
        this.hashMap.clear();
    }

    /**
     * 根据粒子位置计算对应的网格单元格坐标
     */
    getGridIdFromPos(pos) {
        // 将世界坐标转换为单元格索引（向下取整）
        let x = parseInt(pos.x / this.cellSize);
        let y = parseInt(pos.y / this.cellSize);
        // 返回单元格坐标（需提前定义Vector2类，包含x、y属性）
        return new Vector2(x, y);
    }

    /**根据粒子位置计算对应的单元格哈希值*/
    getGridHashFromPos(pos) {
        // 先转换为单元格索引
        let x = parseInt(pos.x / this.cellSize);
        let y = parseInt(pos.y / this.cellSize);
        // 再将索引转为哈希值
        return this.cellIndexToHash(x, y);
    }

    /**
     * 将单元格索引转换为哈希值（核心哈希算法）
     */
    cellIndexToHash(x, y) {
        // 质数参与异或运算混合x/y，取模限制哈希值范围
        let hash = (x * this.prime1 ^ y * this.prime2) % this.hashMapSize;
        return hash;
    }

    getNeighbourOfParticleIdx(i) {
        let neighbours = [];
        let pos = this.particles[i].position;
        let particleGridX = parseInt(pos.x / this.cellSize);
        let particleGridY = parseInt(pos.y / this.cellSize);

        for (let x = -1; x <= 1; x ++ ) {
            for (let y = -1; y <= 1; y ++ ) {
                let gridX = particleGridX + x;
                let gridY = particleGridY + y;

                let hash =  this.cellIndexToHash(gridX, gridY);
                let content = this.getContentOfCell(hash);

                neighbours.push(...content);
            }
        }
        return neighbours;
    }

    /**
     * 遍历所有粒子，将粒子映射到对应的哈希单元格中
     * 核心逻辑：计算粒子哈希值，将粒子存入对应哈希桶的数组
     */
    mapParticleToCell() {
        for (let i = 0; i < this.particles.length; i++) {
            let pos = this.particles[i].position;
            
            let hash = this.getGridHashFromPos(pos);

            let entries = this.hashMap.get(hash);

            // 若单元格无粒子数组，新建数组并存入哈希表
            if (entries == null) {
                let newArray = [this.particles[i]];
                this.hashMap.set(hash, newArray);
            } else {
                // 若已有数组，直接添加粒子
                entries.push(this.particles[i]);
            }
        }
    }

    /**
     * 根据单元格哈希值获取该单元格内的所有粒子
     */
    getContentOfCell(id) {
        let content = this.hashMap.get(id);

        if (content == null) {
            return [];
        } else {
            return content;
        }
    }
}