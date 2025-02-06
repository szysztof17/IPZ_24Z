import {knex} from "knex";

const main = async () => {
    const db = knex({
        client: "mysql2",
        connection: {
            host: "127.0.0.1",
            port: 3306,
            user: "bio",
            password: "bio",
            database: "bio",
        }
    })

    await db.schema.dropTableIfExists("pre_filtered")
    await db.schema.createTable("pre_filtered", builder => {
        builder.integer("cid").notNullable();
        builder.primary(["cid"])
        builder.string("canonicalsmiles").notNullable();
    })

    //select cid, charge, polararea, cmpdname from raw where hbonddonor <= 5 and hbondacc <= 10 and mw <= 500 and xlogp <= 5 and rotbonds <= 10 and polararea <= 140 and charge >= -2 and charge <= 2

    await db.insert(db.select(["cid", "canonicalsmiles"]).from("raw")
        .where('hbonddonor', '<=', 5)
        .andWhere('hbondacc', '<=', 10)
        .andWhere('mw', '<=', 500)
        .andWhere('xlogp', '<=', 10)
        .andWhere('rotbonds', '<=', 10)
        .andWhere('polararea', '<=', 140)
        .andWhere('charge', '>=', -2)
        .andWhere('charge', '<=', 2)
    ).into("pre_filtered")

    await db.destroy()
}

main().then(() => console.log("finished")).catch(console.error)