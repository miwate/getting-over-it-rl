// Flat C API over patched Box2D 2.3.1 for goi_jax/reference/box2d.py (ctypes). Built by scripts/build_box2d.sh.
// Objects are addressed by small integer ids in creation order, so Python never holds raw pointers to bodies,
// fixtures or joints. Everything is float32, as in Box2D; the Python side converts at the boundary only.

// The oracle has to read solver internals (accumulated joint impulses, limit states, sleep timers, inv_dt0) that
// 2.3.1 keeps protected. Opening them up here, in the one translation unit that needs them, keeps the Box2D sources
// themselves stock apart from scripts/patches. Layout is unchanged, so this is ABI-safe against the library objects.
#include <cstdint>
#include <cstring>
#include <vector>  // standard headers first, so the access override below never reaches them

#define protected public
#define private public
#include <Box2D/Box2D.h>
#undef private
#undef protected

// Tuning globals added by scripts/patches/box2d-2.3.1-goi.patch.
#define GOI_PARAMS(X) \
    X(linearSlop, b2_linearSlop) X(angularSlop, b2_angularSlop) X(polygonRadius, b2_polygonRadius) \
    X(velocityThreshold, b2_velocityThreshold) X(maxLinearCorrection, b2_maxLinearCorrection) \
    X(maxAngularCorrection, b2_maxAngularCorrection) X(maxTranslation, b2_maxTranslation) \
    X(maxRotation, b2_maxRotation) X(baumgarte, b2_baumgarte) X(toiBaumgarte, b2_toiBaugarte) \
    X(timeToSleep, b2_timeToSleep) X(linearSleepTolerance, b2_linearSleepTolerance) \
    X(angularSleepTolerance, b2_angularSleepTolerance)

namespace {

// Unity filters pairs with its 32x32 layer matrix, which does not fit 2.3's uint16 category/mask bits, so layers
// live beside the fixtures and this filter replaces the stock group/category test when enabled.
struct Filter : b2ContactFilter
{
    bool useLayers = false;
    uint32 masks[32];
    std::vector<int>* layers = nullptr;  // per fixture id

    bool ShouldCollide(b2Fixture* a, b2Fixture* b) override
    {
        if (!useLayers) return b2ContactFilter::ShouldCollide(a, b);
        int la = (*layers)[(intptr_t)a->GetUserData()], lb = (*layers)[(intptr_t)b->GetUserData()];
        return (masks[la] >> lb) & 1u;
    }
};

// Event log for one step: Begin/End (Unity's Enter/Exit) and PostSolve impulses (from both the main solve and TOI).
struct Event
{
    int32 kind;  // 0 begin, 1 end, 2 post-solve
    int32 fixtureA, fixtureB, childA, childB, count;
    float32 normalImpulses[2], tangentImpulses[2];
};

struct Listener : b2ContactListener
{
    std::vector<Event> events;
    void Log(int kind, b2Contact* c, const b2ContactImpulse* imp)
    {
        Event e = {};
        e.kind = kind;
        e.fixtureA = (int32)(intptr_t)c->GetFixtureA()->GetUserData();
        e.fixtureB = (int32)(intptr_t)c->GetFixtureB()->GetUserData();
        e.childA = c->GetChildIndexA();
        e.childB = c->GetChildIndexB();
        if (imp)
        {
            e.count = imp->count;
            for (int i = 0; i < imp->count; ++i)
            {
                e.normalImpulses[i] = imp->normalImpulses[i];
                e.tangentImpulses[i] = imp->tangentImpulses[i];
            }
        }
        events.push_back(e);
    }
    void BeginContact(b2Contact* c) override { Log(0, c, nullptr); }
    void EndContact(b2Contact* c) override { Log(1, c, nullptr); }
    void PostSolve(b2Contact* c, const b2ContactImpulse* imp) override { Log(2, c, imp); }
};

struct World
{
    b2World world;
    std::vector<b2Body*> bodies;
    std::vector<b2Fixture*> fixtures;
    std::vector<int> layers;
    std::vector<b2Joint*> joints;
    Filter filter;
    Listener listener;

    explicit World(b2Vec2 g) : world(g)
    {
        filter.layers = &layers;
        for (auto& m : filter.masks) m = 0xffffffffu;
        world.SetContactFilter(&filter);
        world.SetContactListener(&listener);
    }
};

bool BodyOk(World* w, int id) { return id >= 0 && id < (int)w->bodies.size(); }
bool FixtureOk(World* w, int id) { return id >= 0 && id < (int)w->fixtures.size(); }
bool JointOk(World* w, int id) { return id >= 0 && id < (int)w->joints.size(); }

int AddFixture(World* w, int body, b2Shape* shape, const float32* mat, const int32* flt)
{
    // mat: friction, restitution, density. flt: sensor, layer, categoryBits, maskBits, groupIndex.
    b2FixtureDef fd;
    fd.shape = shape;
    fd.friction = mat[0];
    fd.restitution = mat[1];
    fd.density = mat[2];  // 0 keeps the explicit b2MassData (2.3.1 only re-derives mass when density > 0)
    fd.isSensor = flt[0] != 0;
    fd.filter.categoryBits = (uint16)flt[2];
    fd.filter.maskBits = (uint16)flt[3];
    fd.filter.groupIndex = (int16)flt[4];
    int id = (int)w->fixtures.size();
    fd.userData = (void*)(intptr_t)id;
    w->layers.push_back(flt[1] & 31);
    w->fixtures.push_back(w->bodies[body]->CreateFixture(&fd));
    return id;
}

}  // namespace

extern "C" {

// ---- process-wide tuning (b2Settings globals are shared by every world in the process)

int goi_b2_set_param(const char* name, float32 value)
{
#define X(n, g) if (std::strcmp(name, #n) == 0) { g = value; return 0; }
    GOI_PARAMS(X)
#undef X
    if (std::strcmp(name, "unityTOI") == 0) { b2_unityTOI = (int32)value; return 0; }
    if (std::strcmp(name, "frictionMix") == 0) { b2_frictionMix = (int32)value; return 0; }
    if (std::strcmp(name, "trigMode") == 0) { b2_trigMode = (int32)value; return 0; }  // scripts/patches/*-trig.patch
    if (std::strcmp(name, "revoluteSolver") == 0) { b2_revoluteSolver = (int32)value; return 0; }  // *-revolute24.patch
    return -1;
}

int goi_b2_get_param(const char* name, float32* value)
{
#define X(n, g) if (std::strcmp(name, #n) == 0) { *value = g; return 0; }
    GOI_PARAMS(X)
#undef X
    if (std::strcmp(name, "unityTOI") == 0) { *value = (float32)b2_unityTOI; return 0; }
    if (std::strcmp(name, "frictionMix") == 0) { *value = (float32)b2_frictionMix; return 0; }
    if (std::strcmp(name, "trigMode") == 0) { *value = (float32)b2_trigMode; return 0; }
    if (std::strcmp(name, "revoluteSolver") == 0) { *value = (float32)b2_revoluteSolver; return 0; }
    return -1;
}

// ---- world

void* goi_b2_world_create(float32 gx, float32 gy) { return new World(b2Vec2(gx, gy)); }
void goi_b2_world_destroy(void* w) { delete (World*)w; }

// flags: allowSleep, warmStarting, continuousPhysics, subStepping, autoClearForces
void goi_b2_world_set_flags(void* wp, const int32* f)
{
    b2World& w = ((World*)wp)->world;
    w.SetAllowSleeping(f[0] != 0);
    w.SetWarmStarting(f[1] != 0);
    w.SetContinuousPhysics(f[2] != 0);
    w.SetSubStepping(f[3] != 0);
    w.SetAutoClearForces(f[4] != 0);
}

void goi_b2_world_set_gravity(void* wp, float32 gx, float32 gy) { ((World*)wp)->world.SetGravity(b2Vec2(gx, gy)); }

// masks: 32 entries of Physics2D.GetLayerCollisionMask; null turns the layer filter off (stock filtering).
// Existing contacts are flagged for re-filtering on the next step, as Box2D does after SetFilterData.
void goi_b2_world_set_layer_masks(void* wp, const uint32* masks)
{
    World* w = (World*)wp;
    w->filter.useLayers = masks != nullptr;
    if (masks) std::memcpy(w->filter.masks, masks, sizeof w->filter.masks);
    for (b2Contact* c = w->world.GetContactList(); c; c = c->GetNext()) c->FlagForFiltering();
}

// Box2D's dtRatio uses the previous step's 1/dt; a fresh world has 0 (no warm-start scaling on its first step).
float32 goi_b2_world_get_inv_dt0(void* wp) { return ((World*)wp)->world.m_inv_dt0; }
void goi_b2_world_set_inv_dt0(void* wp, float32 v) { ((World*)wp)->world.m_inv_dt0 = v; }

void goi_b2_world_step(void* wp, float32 dt, int32 velIters, int32 posIters)
{
    World* w = (World*)wp;
    w->listener.events.clear();
    w->world.Step(dt, velIters, posIters);
}

void goi_b2_world_clear_forces(void* wp) { ((World*)wp)->world.ClearForces(); }

// counts: bodies, fixtures, joints, live contacts, events of the last step
void goi_b2_world_counts(void* wp, int32* out)
{
    World* w = (World*)wp;
    out[0] = (int32)w->bodies.size();
    out[1] = (int32)w->fixtures.size();
    out[2] = (int32)w->joints.size();
    out[3] = w->world.GetContactCount();
    out[4] = (int32)w->listener.events.size();
}

// ---- bodies

// f: x, y, angle, linearDamping, angularDamping, gravityScale. i: type (0 static, 1 kinematic, 2 dynamic),
// fixedRotation, bullet, continuous, awake, allowSleep, active.
int goi_b2_body_add(void* wp, const float32* f, const int32* i)
{
    World* w = (World*)wp;
    b2BodyDef bd;
    bd.type = (b2BodyType)i[0];
    bd.position.Set(f[0], f[1]);
    bd.angle = f[2];
    bd.linearDamping = f[3];
    bd.angularDamping = f[4];
    bd.gravityScale = f[5];
    bd.fixedRotation = i[1] != 0;
    bd.bullet = i[2] != 0;
    bd.awake = i[4] != 0;
    bd.allowSleep = i[5] != 0;
    bd.active = i[6] != 0;
    int id = (int)w->bodies.size();
    bd.userData = (void*)(intptr_t)id;
    b2Body* b = w->world.CreateBody(&bd);
    b->SetContinuous(i[3] != 0);
    w->bodies.push_back(b);
    return id;
}

// Unity gives inertia about the centre of mass; b2MassData wants it about the body origin. SetMassData converts
// back with I - m*|c|^2, which need not round-trip in float32, so exact=1 then stores the given value verbatim
// (open question T7: what Unity itself passes).
int goi_b2_body_set_mass(void* wp, int id, float32 mass, float32 cx, float32 cy, float32 inertiaCom, int32 exact)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    b2MassData md;
    md.mass = mass;
    md.center.Set(cx, cy);
    md.I = inertiaCom + mass * b2Dot(md.center, md.center);
    b->SetMassData(&md);
    if (exact && b->m_type == b2_dynamicBody && b->m_I > 0.0f)
    {
        b->m_I = inertiaCom;
        b->m_invI = 1.0f / inertiaCom;
    }
    return 0;
}

// out (24 floats): originX, originY, angle, centerX, centerY, vx, vy, w, awake, sleepTime, type, active,
// mass, invMass, I (about COM), invI, localCenterX, localCenterY, c0X, c0Y, a0, alpha0, flags, bullet|continuous<<1
int goi_b2_body_get(void* wp, int id, float32* o)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    o[0] = b->m_xf.p.x; o[1] = b->m_xf.p.y; o[2] = b->m_sweep.a;
    o[3] = b->m_sweep.c.x; o[4] = b->m_sweep.c.y;
    o[5] = b->m_linearVelocity.x; o[6] = b->m_linearVelocity.y; o[7] = b->m_angularVelocity;
    o[8] = b->IsAwake(); o[9] = b->m_sleepTime; o[10] = (float32)b->m_type; o[11] = b->IsActive();
    o[12] = b->m_mass; o[13] = b->m_invMass; o[14] = b->m_I; o[15] = b->m_invI;
    o[16] = b->m_sweep.localCenter.x; o[17] = b->m_sweep.localCenter.y;
    o[18] = b->m_sweep.c0.x; o[19] = b->m_sweep.c0.y; o[20] = b->m_sweep.a0; o[21] = b->m_sweep.alpha0;
    o[22] = (float32)b->m_flags; o[23] = (float32)(b->IsBullet() | (b->IsContinuous() << 1));
    return 0;
}

// Teleport the body origin (SetTransform: also re-syncs fixtures; contacts keep their impulses).
int goi_b2_body_set_transform(void* wp, int id, float32 x, float32 y, float32 angle)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    w->bodies[id]->SetTransform(b2Vec2(x, y), angle);
    return 0;
}

// Raw transform and sweep write, e.g. to load a recorded native body bit for bit (Box2D's own setters recompute
// the centre from the origin, which need not round-trip). f: xf.p.x, xf.p.y, xf.q.s, xf.q.c, c0.x, c0.y, c.x, c.y,
// a0, a, alpha0. Fixtures and broad-phase proxies are not re-synced.
int goi_b2_body_set_raw(void* wp, int id, const float32* f)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    b->m_xf.p.Set(f[0], f[1]);
    b->m_xf.q.s = f[2]; b->m_xf.q.c = f[3];
    b->m_sweep.c0.Set(f[4], f[5]);
    b->m_sweep.c.Set(f[6], f[7]);
    b->m_sweep.a0 = f[8]; b->m_sweep.a = f[9]; b->m_sweep.alpha0 = f[10];
    return 0;
}

// Raw velocity write. Box2D's setters ignore static bodies and wake the body when the value is non-zero.
int goi_b2_body_set_velocity(void* wp, int id, float32 vx, float32 vy, float32 av)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    w->bodies[id]->SetLinearVelocity(b2Vec2(vx, vy));
    w->bodies[id]->SetAngularVelocity(av);
    return 0;
}

// awake < 0 / sleepTime < 0 leave that field alone. SetAwake(true) zeroes the sleep timer, so it goes first.
int goi_b2_body_set_sleep(void* wp, int id, int32 awake, float32 sleepTime)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    if (awake >= 0) b->SetAwake(awake != 0);
    if (sleepTime >= 0.0f) b->m_sleepTime = sleepTime;
    return 0;
}

// flags: what = 0 bullet, 1 continuous, 2 active, 3 fixedRotation, 4 allowSleep, 5 type (value = b2BodyType).
// Note SetFixedRotation and SetType call ResetMassData, which drops the explicit mass: set the mass again after.
int goi_b2_body_set_flag(void* wp, int id, int32 what, int32 value)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    switch (what)
    {
    case 0: b->SetBullet(value != 0); break;
    case 1: b->SetContinuous(value != 0); break;
    case 2: b->SetActive(value != 0); break;
    case 3: b->SetFixedRotation(value != 0); break;
    case 4: b->SetSleepingAllowed(value != 0); break;
    case 5: b->SetType((b2BodyType)value); break;
    default: return -1;
    }
    return 0;
}

// f: linearDamping, angularDamping, gravityScale
int goi_b2_body_set_damping(void* wp, int id, float32 ld, float32 ad, float32 gs)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    w->bodies[id]->SetLinearDamping(ld);
    w->bodies[id]->SetAngularDamping(ad);
    w->bodies[id]->SetGravityScale(gs);
    return 0;
}

// Force at the centre of mass plus torque, accumulated until the step clears forces (Rigidbody2D.AddForce).
int goi_b2_body_apply_force(void* wp, int id, float32 fx, float32 fy, float32 torque, int32 wake)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    b->ApplyForceToCenter(b2Vec2(fx, fy), wake != 0);
    if (torque != 0.0f) b->ApplyTorque(torque, wake != 0);
    return 0;
}

// Kinematic MovePosition emulation: the velocity that reaches the target origin pose in one step of dt.
// Unity's exact rule (and what a same-step velocity write does to it) is open question T3.
int goi_b2_body_move_to(void* wp, int id, float32 x, float32 y, float32 angle, float32 dt)
{
    World* w = (World*)wp;
    if (!BodyOk(w, id)) return -1;
    b2Body* b = w->bodies[id];
    float32 inv = 1.0f / dt;
    b->SetLinearVelocity(b2Vec2((x - b->m_xf.p.x) * inv, (y - b->m_xf.p.y) * inv));
    b->SetAngularVelocity((angle - b->m_sweep.a) * inv);
    return 0;
}

// ---- fixtures (mat: friction, restitution, density; flt: sensor, layer, categoryBits, maskBits, groupIndex)

// raw = 0: b2PolygonShape::Set (convex hull from the rightmost vertex, CCW). Note 2.3.1's Set welds points closer
// than sqrt(0.5 * b2_linearSlop) = 5 cm (it compares a squared distance with a length), which destroys
// centimetre-scale pieces such as the hammer tip's; Unity cannot be using it as-is for those.
// raw = 1: take the vertices verbatim (must be strictly convex and CCW, e.g. pieces read back from Unity's native
// shapes) and derive the normals exactly as Set does. The centroid only feeds mass, which is explicit here.
int goi_b2_fixture_add_polygon(void* wp, int body, const float32* xy, int32 n, float32 radius, int32 raw,
                               const float32* mat, const int32* flt)
{
    World* w = (World*)wp;
    if (!BodyOk(w, body) || n < 3 || n > b2_maxPolygonVertices) return -1;
    b2Vec2 v[b2_maxPolygonVertices];
    for (int i = 0; i < n; ++i) v[i].Set(xy[2 * i], xy[2 * i + 1]);
    b2PolygonShape s;
    if (raw)
    {
        s.m_count = n;
        s.m_centroid.SetZero();
        for (int i = 0; i < n; ++i)
        {
            b2Vec2 edge = v[i + 1 < n ? i + 1 : 0] - v[i];
            // Every other vertex strictly left of every edge: a neighbour-only test lets a self-crossing
            // (doubly wound) pentagon through.
            for (int k = 2; k < n; ++k)
                if (b2Cross(edge, v[(i + k) % n] - v[i]) <= 0.0f) return -3;  // not strictly convex CCW
            s.m_vertices[i] = v[i];
            s.m_normals[i] = b2Cross(edge, 1.0f);
            s.m_normals[i].Normalize();
            s.m_centroid += (1.0f / n) * v[i];
        }
    }
    else
    {
        s.Set(v, n);
        // Set falls back to SetAsBox(1, 1) when welding leaves < 3 points: every hull vertex must be an input.
        for (int k = 0; k < s.m_count; ++k)
        {
            bool found = false;
            for (int i = 0; i < n && !found; ++i) found = s.m_vertices[k].x == v[i].x && s.m_vertices[k].y == v[i].y;
            if (!found) return -2;
        }
    }
    s.m_radius = radius;
    return AddFixture(w, body, &s, mat, flt);
}

int goi_b2_fixture_add_box(void* wp, int body, float32 hx, float32 hy, float32 cx, float32 cy, float32 angle,
                           float32 radius, const float32* mat, const int32* flt)
{
    World* w = (World*)wp;
    if (!BodyOk(w, body)) return -1;
    b2PolygonShape s;
    s.SetAsBox(hx, hy, b2Vec2(cx, cy), angle);
    s.m_radius = radius;
    return AddFixture(w, body, &s, mat, flt);
}

int goi_b2_fixture_add_circle(void* wp, int body, float32 cx, float32 cy, float32 radius,
                              const float32* mat, const int32* flt)
{
    World* w = (World*)wp;
    if (!BodyOk(w, body)) return -1;
    b2CircleShape s;
    s.m_p.Set(cx, cy);
    s.m_radius = radius;
    return AddFixture(w, body, &s, mat, flt);
}

// Box2D mixes friction once, when a contact is created. refresh = 1 also re-mixes it on this fixture's existing
// contacts (b2Contact::ResetFriction), one candidate for what Unity does on a sharedMaterial swap (T5).
int goi_b2_fixture_set_friction(void* wp, int id, float32 friction, int32 refresh)
{
    World* w = (World*)wp;
    if (!FixtureOk(w, id)) return -1;
    b2Fixture* f = w->fixtures[id];
    f->SetFriction(friction);
    if (refresh)
        for (b2ContactEdge* ce = f->GetBody()->GetContactList(); ce; ce = ce->next)
            if (ce->contact->GetFixtureA() == f || ce->contact->GetFixtureB() == f) ce->contact->ResetFriction();
    return 0;
}

int goi_b2_fixture_set_restitution(void* wp, int id, float32 r)
{
    World* w = (World*)wp;
    if (!FixtureOk(w, id)) return -1;
    w->fixtures[id]->SetRestitution(r);
    return 0;
}

// flt as for add (sensor, layer, categoryBits, maskBits, groupIndex); re-filters this fixture's contacts.
int goi_b2_fixture_set_filter(void* wp, int id, const int32* flt)
{
    World* w = (World*)wp;
    if (!FixtureOk(w, id)) return -1;
    b2Fixture* f = w->fixtures[id];
    f->SetSensor(flt[0] != 0);
    w->layers[id] = flt[1] & 31;
    b2Filter fd;
    fd.categoryBits = (uint16)flt[2];
    fd.maskBits = (uint16)flt[3];
    fd.groupIndex = (int16)flt[4];
    f->SetFilterData(fd);  // flags the contacts for filtering, which consults the layer too
    return 0;
}

// out (8): body, type (0 circle, 2 polygon, as b2Shape::Type), childCount, vertexCount, radius, friction,
// restitution, layer. verts: up to 8 local points (circle: its centre).
int goi_b2_fixture_get(void* wp, int id, float32* o, float32* verts)
{
    World* w = (World*)wp;
    if (!FixtureOk(w, id)) return -1;
    b2Fixture* f = w->fixtures[id];
    b2Shape* s = f->GetShape();
    o[0] = (float32)(intptr_t)f->GetBody()->GetUserData();
    o[1] = (float32)s->GetType();
    o[2] = (float32)s->GetChildCount();
    o[4] = s->m_radius;
    o[5] = f->GetFriction();
    o[6] = f->GetRestitution();
    o[7] = (float32)w->layers[id];
    if (s->GetType() == b2Shape::e_polygon)
    {
        b2PolygonShape* p = (b2PolygonShape*)s;
        o[3] = (float32)p->m_count;
        for (int i = 0; i < p->m_count; ++i) { verts[2 * i] = p->m_vertices[i].x; verts[2 * i + 1] = p->m_vertices[i].y; }
    }
    else if (s->GetType() == b2Shape::e_circle)
    {
        o[3] = 1;
        verts[0] = ((b2CircleShape*)s)->m_p.x;
        verts[1] = ((b2CircleShape*)s)->m_p.y;
    }
    else o[3] = 0;
    return 0;
}

// Full shape data, for checking another engine's shape construction bit for bit. out (36): vertexCount, radius,
// centroid x, y, then 8 vertices (x, y), then 8 normals (x, y). A circle reports count 1 and its centre as vertex 0.
int goi_b2_fixture_shape(void* wp, int id, float32* o)
{
    World* w = (World*)wp;
    if (!FixtureOk(w, id)) return -1;
    b2Shape* s = w->fixtures[id]->GetShape();
    std::memset(o, 0, 36 * sizeof(float32));
    o[1] = s->m_radius;
    if (s->GetType() == b2Shape::e_polygon)
    {
        b2PolygonShape* p = (b2PolygonShape*)s;
        o[0] = (float32)p->m_count;
        o[2] = p->m_centroid.x; o[3] = p->m_centroid.y;
        for (int i = 0; i < p->m_count; ++i)
        {
            o[4 + 2 * i] = p->m_vertices[i].x; o[5 + 2 * i] = p->m_vertices[i].y;
            o[20 + 2 * i] = p->m_normals[i].x; o[21 + 2 * i] = p->m_normals[i].y;
        }
    }
    else if (s->GetType() == b2Shape::e_circle)
    {
        b2CircleShape* c = (b2CircleShape*)s;
        o[0] = 1; o[4] = c->m_p.x; o[5] = c->m_p.y;
    }
    return 0;
}

// ---- joints. Common: bodyA, bodyB, collideConnected; anchors are local to each body's origin.

// f: laX, laY, lbX, lbY, referenceAngle, lower, upper, motorSpeed, maxMotorTorque. i: enableLimit, enableMotor
int goi_b2_joint_add_revolute(void* wp, int a, int b, int32 collide, const float32* f, const int32* i)
{
    World* w = (World*)wp;
    if (!BodyOk(w, a) || !BodyOk(w, b)) return -1;
    b2RevoluteJointDef d;
    d.bodyA = w->bodies[a]; d.bodyB = w->bodies[b]; d.collideConnected = collide != 0;
    d.localAnchorA.Set(f[0], f[1]); d.localAnchorB.Set(f[2], f[3]);
    d.referenceAngle = f[4]; d.lowerAngle = f[5]; d.upperAngle = f[6];
    d.motorSpeed = f[7]; d.maxMotorTorque = f[8];
    d.enableLimit = i[0] != 0; d.enableMotor = i[1] != 0;
    w->joints.push_back(w->world.CreateJoint(&d));
    return (int)w->joints.size() - 1;
}

// f: laX, laY, lbX, lbY, axisX, axisY (local to A, normalized by Box2D), referenceAngle, lower, upper,
// motorSpeed, maxMotorForce. i: enableLimit, enableMotor
int goi_b2_joint_add_prismatic(void* wp, int a, int b, int32 collide, const float32* f, const int32* i)
{
    World* w = (World*)wp;
    if (!BodyOk(w, a) || !BodyOk(w, b)) return -1;
    b2PrismaticJointDef d;
    d.bodyA = w->bodies[a]; d.bodyB = w->bodies[b]; d.collideConnected = collide != 0;
    d.localAnchorA.Set(f[0], f[1]); d.localAnchorB.Set(f[2], f[3]);
    d.localAxisA.Set(f[4], f[5]); d.localAxisA.Normalize();
    d.referenceAngle = f[6]; d.lowerTranslation = f[7]; d.upperTranslation = f[8];
    d.motorSpeed = f[9]; d.maxMotorForce = f[10];
    d.enableLimit = i[0] != 0; d.enableMotor = i[1] != 0;
    w->joints.push_back(w->world.CreateJoint(&d));
    return (int)w->joints.size() - 1;
}

// f: laX, laY, lbX, lbY, referenceAngle, frequencyHz, dampingRatio (frequency 0 = rigid)
int goi_b2_joint_add_weld(void* wp, int a, int b, int32 collide, const float32* f)
{
    World* w = (World*)wp;
    if (!BodyOk(w, a) || !BodyOk(w, b)) return -1;
    b2WeldJointDef d;
    d.bodyA = w->bodies[a]; d.bodyB = w->bodies[b]; d.collideConnected = collide != 0;
    d.localAnchorA.Set(f[0], f[1]); d.localAnchorB.Set(f[2], f[3]);
    d.referenceAngle = f[4]; d.frequencyHz = f[5]; d.dampingRatio = f[6];
    w->joints.push_back(w->world.CreateJoint(&d));
    return (int)w->joints.size() - 1;
}

// what: 0 motorSpeed, 1 maxMotorTorque/Force, 2 enableMotor, 3 enableLimit, 4 limits (value, value2),
// 5 weld frequency, 6 weld damping. Setters wake both bodies, as in Box2D.
int goi_b2_joint_set(void* wp, int id, int32 what, float32 value, float32 value2)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    if (j->GetType() == e_revoluteJoint)
    {
        b2RevoluteJoint* r = (b2RevoluteJoint*)j;
        switch (what)
        {
        case 0: r->SetMotorSpeed(value); return 0;
        case 1: r->SetMaxMotorTorque(value); return 0;
        case 2: r->EnableMotor(value != 0.0f); return 0;
        case 3: r->EnableLimit(value != 0.0f); return 0;
        case 4: r->SetLimits(value, value2); return 0;
        }
    }
    else if (j->GetType() == e_prismaticJoint)
    {
        b2PrismaticJoint* p = (b2PrismaticJoint*)j;
        switch (what)
        {
        case 0: p->SetMotorSpeed(value); return 0;
        case 1: p->SetMaxMotorForce(value); return 0;
        case 2: p->EnableMotor(value != 0.0f); return 0;
        case 3: p->EnableLimit(value != 0.0f); return 0;
        case 4: p->SetLimits(value, value2); return 0;
        }
    }
    else if (j->GetType() == e_weldJoint)
    {
        b2WeldJoint* d = (b2WeldJoint*)j;
        switch (what)
        {
        case 5: d->SetFrequency(value); return 0;
        case 6: d->SetDampingRatio(value); return 0;
        }
    }
    return -2;
}

// out (16): type (b2JointType), bodyA, bodyB, reactionForceX, reactionForceY, reactionTorque (both at inv_dt),
// impulseX, impulseY, impulseZ, motorImpulse, limitState, value (angle / translation), speed,
// motorSpeed, maxMotor, extra (weld: gamma; others 0). Weld bias goes in out[16]; revolute lowerImpulse and
// upperImpulse (v2.4 solver, scripts/patches/*-revolute24.patch) in out[17], out[18].
int goi_b2_joint_get(void* wp, int id, float32 inv_dt, float32* o)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    std::memset(o, 0, 19 * sizeof(float32));
    o[0] = (float32)j->GetType();
    o[1] = (float32)(intptr_t)j->GetBodyA()->GetUserData();
    o[2] = (float32)(intptr_t)j->GetBodyB()->GetUserData();
    b2Vec2 rf = j->GetReactionForce(inv_dt);
    o[3] = rf.x; o[4] = rf.y; o[5] = j->GetReactionTorque(inv_dt);
    if (j->GetType() == e_revoluteJoint)
    {
        b2RevoluteJoint* r = (b2RevoluteJoint*)j;
        o[6] = r->m_impulse.x; o[7] = r->m_impulse.y; o[8] = r->m_impulse.z; o[9] = r->m_motorImpulse;
        o[10] = (float32)r->m_limitState; o[11] = r->GetJointAngle(); o[12] = r->GetJointSpeed();
        o[13] = r->m_motorSpeed; o[14] = r->m_maxMotorTorque;
        o[17] = r->m_lowerImpulse; o[18] = r->m_upperImpulse;
    }
    else if (j->GetType() == e_prismaticJoint)
    {
        b2PrismaticJoint* p = (b2PrismaticJoint*)j;
        o[6] = p->m_impulse.x; o[7] = p->m_impulse.y; o[8] = p->m_impulse.z; o[9] = p->m_motorImpulse;
        o[10] = (float32)p->m_limitState; o[11] = p->GetJointTranslation(); o[12] = p->GetJointSpeed();
        o[13] = p->m_motorSpeed; o[14] = p->m_maxMotorForce;
    }
    else if (j->GetType() == e_weldJoint)
    {
        b2WeldJoint* d = (b2WeldJoint*)j;
        o[6] = d->m_impulse.x; o[7] = d->m_impulse.y; o[8] = d->m_impulse.z;
        o[13] = d->m_frequencyHz; o[14] = d->m_dampingRatio; o[15] = d->m_gamma; o[16] = d->m_bias;
    }
    return 0;
}

// Overwrite accumulated joint impulses (warm start state), e.g. to replay a recorded mid-trajectory step.
int goi_b2_joint_set_impulses(void* wp, int id, float32 x, float32 y, float32 z, float32 motor)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    if (j->GetType() == e_revoluteJoint) { auto* r = (b2RevoluteJoint*)j; r->m_impulse.Set(x, y, z); r->m_motorImpulse = motor; }
    else if (j->GetType() == e_prismaticJoint) { auto* p = (b2PrismaticJoint*)j; p->m_impulse.Set(x, y, z); p->m_motorImpulse = motor; }
    else if (j->GetType() == e_weldJoint) ((b2WeldJoint*)j)->m_impulse.Set(x, y, z);
    else return -2;
    return 0;
}

// Revolute lower/upper limit impulses (v2.4 solver state; the v2.3.1 solver keeps its limit impulse in impulse.z).
int goi_b2_joint_set_limit_impulses(void* wp, int id, float32 lower, float32 upper)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    if (j->GetType() != e_revoluteJoint) return -2;
    auto* r = (b2RevoluteJoint*)j;
    r->m_lowerImpulse = lower; r->m_upperImpulse = upper;
    return 0;
}

// The joint frame as stored (out 12): localAnchorA x, y, localAnchorB x, y, referenceAngle, local axis x, y and
// perpendicular x, y (prismatic localXAxisA / localYAxisA; others 0), lower, upper (angle or translation; weld 0),
// collideConnected.
int goi_b2_joint_frame(void* wp, int id, float32* o)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    std::memset(o, 0, 12 * sizeof(float32));
    o[11] = (float32)j->m_collideConnected;
    if (j->GetType() == e_revoluteJoint)
    {
        auto* r = (b2RevoluteJoint*)j;
        o[0] = r->m_localAnchorA.x; o[1] = r->m_localAnchorA.y; o[2] = r->m_localAnchorB.x; o[3] = r->m_localAnchorB.y;
        o[4] = r->m_referenceAngle; o[9] = r->m_lowerAngle; o[10] = r->m_upperAngle;
    }
    else if (j->GetType() == e_prismaticJoint)
    {
        auto* p = (b2PrismaticJoint*)j;
        o[0] = p->m_localAnchorA.x; o[1] = p->m_localAnchorA.y; o[2] = p->m_localAnchorB.x; o[3] = p->m_localAnchorB.y;
        o[4] = p->m_referenceAngle; o[5] = p->m_localXAxisA.x; o[6] = p->m_localXAxisA.y;
        o[7] = p->m_localYAxisA.x; o[8] = p->m_localYAxisA.y; o[9] = p->m_lowerTranslation; o[10] = p->m_upperTranslation;
    }
    else if (j->GetType() == e_weldJoint)
    {
        auto* d = (b2WeldJoint*)j;
        o[0] = d->m_localAnchorA.x; o[1] = d->m_localAnchorA.y; o[2] = d->m_localAnchorB.x; o[3] = d->m_localAnchorB.y;
        o[4] = d->m_referenceAngle;
    }
    else return -2;
    return 0;
}

// Solver temporaries of the joint's last InitVelocityConstraints (out 14): rA x, y, rB x, y, point mass K ex.x,
// ex.y, ey.x, ey.y, angle, axial mass, lowerImpulse, upperImpulse, indexA, indexB. Revolute v2.4: m_K (not
// inverted), m_angle, m_axialMass (the fields Unity's native joint holds); revolute v2.3.1: m_mass's 2x2 block, the
// joint angle at the step start is not stored (0), m_motorMass. Weld: rA, rB and m_mass's 2x2 block. Prismatic:
// indices only (it has no rA/rB).
int goi_b2_joint_solver(void* wp, int id, float32* o)
{
    World* w = (World*)wp;
    if (!JointOk(w, id)) return -1;
    b2Joint* j = w->joints[id];
    std::memset(o, 0, 14 * sizeof(float32));
    if (j->GetType() == e_revoluteJoint)
    {
        auto* r = (b2RevoluteJoint*)j;
        o[0] = r->m_rA.x; o[1] = r->m_rA.y; o[2] = r->m_rB.x; o[3] = r->m_rB.y;
        if (b2_revoluteSolver)
        {
            o[4] = r->m_K.ex.x; o[5] = r->m_K.ex.y; o[6] = r->m_K.ey.x; o[7] = r->m_K.ey.y;
            o[8] = r->m_angle; o[9] = r->m_axialMass;
        }
        else
        {
            o[4] = r->m_mass.ex.x; o[5] = r->m_mass.ex.y; o[6] = r->m_mass.ey.x; o[7] = r->m_mass.ey.y;
            o[9] = r->m_motorMass;
        }
        o[10] = r->m_lowerImpulse; o[11] = r->m_upperImpulse;
        o[12] = (float32)r->m_indexA; o[13] = (float32)r->m_indexB;
    }
    else if (j->GetType() == e_weldJoint)
    {
        auto* d = (b2WeldJoint*)j;
        o[0] = d->m_rA.x; o[1] = d->m_rA.y; o[2] = d->m_rB.x; o[3] = d->m_rB.y;
        o[4] = d->m_mass.ex.x; o[5] = d->m_mass.ex.y; o[6] = d->m_mass.ey.x; o[7] = d->m_mass.ey.y;
        o[12] = (float32)d->m_indexA; o[13] = (float32)d->m_indexB;
    }
    else if (j->GetType() == e_prismaticJoint)
    {
        auto* p = (b2PrismaticJoint*)j;
        o[12] = (float32)p->m_indexA; o[13] = (float32)p->m_indexB;
    }
    else return -2;
    return 0;
}

// ---- contacts, in world contact-list order (newest first). Not the solve order: b2World::Solve builds each island
// by walking every body's own contact edges depth-first from the body list, so the solver order follows from that.

struct GoiContact
{
    int32 fixtureA, fixtureB, childA, childB, bodyA, bodyB;
    int32 flags;  // b2Contact::m_flags: 1 island, 2 touching, 4 enabled, 8 filter, 16 bullet hit, 32 toi
    int32 manifoldType, pointCount, toiCount;
    float32 friction, restitution, tangentSpeed, toi;
    float32 localNormal[2], localPoint[2];
    float32 pointLocal[2][2], normalImpulse[2], tangentImpulse[2];
    uint32 key[2];
    float32 worldNormal[2], worldPoints[2][2], separations[2];
};

int32 goi_b2_contact_struct_size() { return (int32)sizeof(GoiContact); }

// Fills up to cap records; returns the number of live contacts (call again with a bigger buffer if > cap).
int32 goi_b2_contacts(void* wp, GoiContact* out, int32 cap)
{
    World* w = (World*)wp;
    int32 n = 0;
    for (b2Contact* c = w->world.GetContactList(); c; c = c->GetNext(), ++n)
    {
        if (n >= cap) continue;
        GoiContact& r = out[n];
        std::memset(&r, 0, sizeof r);
        b2Fixture* fa = c->GetFixtureA();
        b2Fixture* fb = c->GetFixtureB();
        r.fixtureA = (int32)(intptr_t)fa->GetUserData();
        r.fixtureB = (int32)(intptr_t)fb->GetUserData();
        r.childA = c->GetChildIndexA();
        r.childB = c->GetChildIndexB();
        r.bodyA = (int32)(intptr_t)fa->GetBody()->GetUserData();
        r.bodyB = (int32)(intptr_t)fb->GetBody()->GetUserData();
        r.flags = (int32)c->m_flags;
        r.toiCount = c->m_toiCount;
        r.friction = c->m_friction;
        r.restitution = c->m_restitution;
        r.tangentSpeed = c->m_tangentSpeed;
        r.toi = c->m_toi;
        const b2Manifold& m = c->m_manifold;
        r.manifoldType = (int32)m.type;
        r.pointCount = m.pointCount;
        r.localNormal[0] = m.localNormal.x; r.localNormal[1] = m.localNormal.y;
        r.localPoint[0] = m.localPoint.x; r.localPoint[1] = m.localPoint.y;
        for (int i = 0; i < m.pointCount; ++i)
        {
            r.pointLocal[i][0] = m.points[i].localPoint.x;
            r.pointLocal[i][1] = m.points[i].localPoint.y;
            r.normalImpulse[i] = m.points[i].normalImpulse;
            r.tangentImpulse[i] = m.points[i].tangentImpulse;
            r.key[i] = m.points[i].id.key;
        }
        if (m.pointCount > 0)
        {
            b2WorldManifold wm;
            c->GetWorldManifold(&wm);
            r.worldNormal[0] = wm.normal.x; r.worldNormal[1] = wm.normal.y;
            for (int i = 0; i < m.pointCount; ++i)
            {
                r.worldPoints[i][0] = wm.points[i].x;
                r.worldPoints[i][1] = wm.points[i].y;
                r.separations[i] = wm.separations[i];
            }
        }
    }
    return n;
}

int32 goi_b2_event_struct_size() { return (int32)sizeof(Event); }

// Begin/End/PostSolve events of the last step, in the order Box2D raised them.
int32 goi_b2_events(void* wp, Event* out, int32 cap)
{
    World* w = (World*)wp;
    int32 n = (int32)w->listener.events.size();
    for (int32 i = 0; i < n && i < cap; ++i) out[i] = w->listener.events[i];
    return n;
}

}  // extern "C"
