import {knex} from "knex";

type Ligand = {
    cid: number
    canonicalSmiles: string
}

const ligands: Ligand[] = [
    {
        cid: 139030486,
        canonicalSmiles: "C1=CC(=CC2=C1C=CN2)C3=C(C#N)C(=NN3)N"
    },
    {
        cid: 139030491,
        canonicalSmiles: "C1=CN=CC(=C1)CN2C=C(C3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N)Cl"
    },
    {
        cid: 145946018,
        canonicalSmiles: "CN1CCCC[C@@H]1CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 139030495,
        canonicalSmiles: "C1CCN(CC1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=C(C#N)C(=NN5)N"
    },
    {
        cid: 139030496,
        canonicalSmiles: "C1=C(C=CC(=C1)CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N)CN5CCOCC5"
    },
    {
        cid: 145997887,
        canonicalSmiles: "C1=CC=C(C=C1)CN2CCC[C@H](C2)CN3C=CC4=C3C=C(C=C4)C5=C(C#N)C(=NN5)N"
    },
    {
        cid: 139030501,
        canonicalSmiles: "COC1=CC(=CC=C1)COC(=O)C2=CNN=C2"
    },
    {
        cid: 68910199,
        canonicalSmiles: "C1=CC=C(C=C1)COC(=O)C2=CNN=C2"
    },
    {
        cid: 139030484,
        canonicalSmiles: "COC1=CC(=CC(=C1)COC(=O)C2=CNN=C2)C3=CN=CC=C3"
    },
    {
        cid: 139030485,
        canonicalSmiles: "COC1=CC=C(C=C1)CN2C=CC3=C2C=C(C=C3)C4=CC(=NN4)N"
    },
    {
        cid: 139030487,
        canonicalSmiles: "C1=CC(=C(C=C1)CN2C=CC3=C2C=C(C=C3)C4=CC(=NN4)N)C#N"
    },
    {
        cid: 139030488,
        canonicalSmiles: "C1=CC=C(C=C1)CN2C=CC3=C2C=C(C=C3)C4=CC(=NN4)N"
    },
    {
        cid: 139030489,
        canonicalSmiles: "C1=CN=CC(=C1)CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 139030490,
        canonicalSmiles: "C1=CC(=CC2=C1C=CN2CC3=CC=NC=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 139030492,
        canonicalSmiles: "C1=CNC(=O)C(=C1)CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 139030493,
        canonicalSmiles: "CN1CCC[C@@H](C1)CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 139030497,
        canonicalSmiles: "C1CCN(CC1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=CC(=NN5)N"
    },
    {
        cid: 139030498,
        canonicalSmiles: "COC1=CC(=CC=C1)CN2C=CC3=C2C=C(C=C3)C4=CC(=NN4)N"
    },
    {
        cid: 139030500,
        canonicalSmiles: "CC(C)N1CCN(CC1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=C(C(=NN5)N)C#N"
    },
    {
        cid: 139030502,
        canonicalSmiles: "COC(=O)COC1=CC(=CC=C1)COC(=O)C2=CNN=C2"
    },
    {
        cid: 2819674,
        canonicalSmiles: "C1=CC(=CC(=C1)CSC2=CC=CS2)C(=O)O"
    },
    {
        //irritant!
        cid: 2763205,
        canonicalSmiles: "C1=CC(=CC2=C1C=CN2)B(O)O"
    },
    {
        cid: 21362592,
        canonicalSmiles: "C1=CC=C(C=C1)C2=NC=C(C=C2)CN"
    },
    {
        cid: 145927357,
        canonicalSmiles: "COC(=O)C1=CC2=C(N1)SC(=C2)CO"
    },
    {
        cid: 117287823,
        canonicalSmiles: "C1=CC(=CC2=C1C=CN2)C3=CC(=NN3)N"
    },
    {
        cid: 145997883,
        canonicalSmiles: "C1=CC(=NC=C1)CN2C=CC3=C2C=C(C=C3)C4=C(C#N)C(=NN4)N"
    },
    {
        cid: 145997884,
        canonicalSmiles: "C1CCN(C1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=CC(=NN5)N"
    },
    {
        //corrosive, irritant!
        cid: 523184,
        canonicalSmiles: "C1=CSC(=C1)C2=CC(=NN2)N"
    },
    {
        //irritant!
        cid: 2799515,
        canonicalSmiles: "C1=C(C(=S)N)ON=C1"
    },
    {
        //irritant!
        cid: 2756469,
        canonicalSmiles: "COC1=CC=C(C=C1)C2=CC(=NN2)N"
    },
    {
        cid: 596767,
        canonicalSmiles: "COC1=CC2=C(C=C1)N=C(C(=O)O)S2"
    },
    {
        cid: 145997882,
        canonicalSmiles: "C1=CC(=NC=C1)CN2C=CC3=C2C=C(C=C3)C4=CC(=NN4)N"
    },
    {
        cid: 145997885,
        canonicalSmiles: "C1CCN(C1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=C(C#N)C(=NN5)N"
    },
    {
        cid: 145997886,
        canonicalSmiles: "CN1CCN(CC1)CC2=CC=C(C=C2)CN3C=CC4=C3C=C(C=C4)C5=C(C#N)C(=NN5)N"
    },
    {
        //irritant!
        cid: 65482,
        canonicalSmiles: "C(C[C@@H](C(=O)O)N)[C@@H](C[C@@H]1[C@H]([C@H]([C@H](N2C=NC3=C2N=CN=C3N)O1)O)O)N"
    },
    {
        cid: 137347758,
        canonicalSmiles: "CCN(CC)CC1=CC=C(C=C1)CNC(=O)C2=CSC3=C2C(=O)NC=N3"
    },
    {
        cid: 137347757,
        canonicalSmiles: "CCCCCCCCNCC1=CC=C(C=C1)CNC(=O)C2=CSC3=C2C(=O)NC=N3"
    },
    {
        cid: 97429072,
        canonicalSmiles: "C1=CC2=C(C=C1)NC(=C2)C(=O)N[C@H]3CCCN(C3)C4=NNC(=C4)C5=CC=NC=C5"
    },
    {
        cid: 71724899,
        canonicalSmiles: "C1=C(C=CC(=C1)CN2CCC(CC2)N)CNC(=O)C3=CSC4=C3C(=O)NC=N4"
    }
]

async function main() {
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

    await db.schema.dropTableIfExists("ligands")
    await db.schema.createTable("ligands", builder => {
        builder.integer("cid").notNullable();
        builder.text("canonicalsmiles").notNullable()
        builder.primary(["cid"])
    })

    for (const ligand of ligands) {
        await db('ligands').insert({
            cid: ligand.cid,
            canonicalsmiles: ligand.canonicalSmiles
        })
    }

    await db.destroy()
}

main().then(() => console.log("finished")).catch(console.error);