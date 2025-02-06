import {knex} from "knex";

const calcLINGOSim = (smilesA: string, smilesB: string, q: number): number =>
{
    const lingoA = findLINGOs(smilesA, q);
    const lingoB = findLINGOs(smilesB, q);

    return innerCalcLINGOSim(lingoA, lingoB);
}

const innerCalcLINGOSim = (lingoArray1: Map<string, number>, lingoArray2: Map<string, number>): number =>
{
    let sim = 0;
    let shared = 0;
    for (const [key, value] of lingoArray1) {
        const freq1 = value;
        let freq2 = 0;

        if (lingoArray2.has(key)) {
            freq2 = lingoArray2.get(key)!!;
            shared = shared + 1;
        }

        sim += 1 - (Math.abs(freq1 - freq2) / Math.abs(freq1 + freq2));
    }

    const denom = lingoArray1.size + lingoArray2.size - shared;
    return sim / denom;
}


const findLINGOs = (smiles: string, q: number): Map<string, number> => {
    const lingoList = new Map<string, number>();

    let tmpSmiles = normalizeSmiles(smiles);

    if (tmpSmiles.length < q) {
        while (tmpSmiles.length != q)
            tmpSmiles = tmpSmiles + "J";
    }

    for (let i = 0; i < tmpSmiles.length - (q - 1); i++) {
        const lingo = smiles.substring(i, i + q);

        if (!lingoList.has(lingo))
            lingoList.set(lingo, 1);
        else {
            const freq = lingoList.get(lingo)!!;
            lingoList.delete(lingo);
            lingoList.set(lingo, freq + 1);
        }
    }

    return lingoList;
}

const normalizeSmiles = (smiles: string) =>
    smiles.replaceAll("Br", "R").replaceAll("Cl", "L").replaceAll("[1-9]", "0")

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

    await db.schema.dropTableIfExists("sim")
    await db.schema.createTable("sim", builder => {
        builder.integer("cid").notNullable();
        builder.float("sim").notNullable()
        builder.integer("target").notNullable();
        builder.primary(["cid", "target"])
    })

    type RawData = {
        cid: number
        canonicalsmiles: string
    }

    let i = 0;

    const smiles: RawData[] = await db("pre_filtered").select(["cid", "canonicalsmiles"])

    console.log("starting")

    const target = smiles[0]

    for (const element of smiles) {
        ++i
        if (i%10000 == 0)
            console.log(i)

        const sim = calcLINGOSim(element.canonicalsmiles, target.canonicalsmiles, 3)
        await db("sim").insert({ cid: element.cid, target: target.cid, sim })
    }

    await db.destroy()
}

main().then(() => console.log("finished")).catch(console.error);